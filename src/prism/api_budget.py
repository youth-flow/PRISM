"""Local shared USD reservations and recoverable API dispatches.

Unknown/failed dispatches retain their full reservation. A completed response is
saved before the caller's JSONL append, so replay after a crash never resends it.
Prices are an explicit snapshot, not a substitute for the provider invoice.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.error import URLError


# USD / million tokens: ordinary, cache read, 30m write, 12h write, output.
PRICES = {"gpt-6.1-sol": (2, .1, 2.5, 4, 10),
          "gpt-6-luna": (.1, .01, .125, .2, .5)}
PRICE_DATE = "2026-11-16"


class BudgetExceeded(RuntimeError):
    def __init__(self, message, *, waitable=False):
        super().__init__(message)
        self.waitable = waitable


def micro_usd(value):
    return int((Decimal(str(value))*1_000_000).to_integral_value(rounding=ROUND_CEILING))


def usage_micro_usd(model, usage, service_tier=None):
    if service_tier not in (None, "auto", "default", "flex"):
        raise ValueError("Unknown billing tier; do not assume a discounted price")
    ordinary, read_rate, write30_rate, write12_rate, output_rate = map(Decimal, map(str, PRICES[model]))
    details = usage.get("input_tokens_details", {})
    cached = details.get("cached_tokens", 0)
    writes = details.get("cache_write_tokens", 0)
    split = details.get("cache_write_tokens_details", {})
    w12 = split.get("ttl_12h_tokens", 0)
    w30 = split.get("ttl_30m_tokens", writes-w12)
    total, output = usage["input_tokens"], usage["output_tokens"]
    numbers = (total, output, cached, writes, w12, w30)
    if any(type(n) is not int or n < 0 for n in numbers) or cached+writes > total or w12+w30 != writes:
        raise ValueError("Inconsistent provider token accounting")
    # Rates per million USD equal micro-USD per token. Round per request upward.
    amount = ((total-cached-writes)*ordinary + cached*read_rate + w30*write30_rate
              + w12*write12_rate + output*output_rate)
    if service_tier == "flex":
        amount *= Decimal("0.5")
    return int(amount.to_integral_value(rounding=ROUND_CEILING))


def request_reservation(model, payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # Conservative byte-bound allowance for template/protocol tokens; the output
    # cap includes reasoning. The settlement checks this engineering bound.
    input_upper = 2*len(encoded)+4096
    output_upper = payload["max_output_tokens"]
    input_rate = max(PRICES[model][:4])
    return int(Decimal(str(input_rate))*input_upper + Decimal(str(PRICES[model][4]))*output_upper)+1


class USDBudget:
    def __init__(self, path, *, total_usd, purposes):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = {"total": micro_usd(total_usd), **{k: micro_usd(v) for k,v in purposes.items()}}
        if any(v <= 0 for v in self.limits.values()) or "total" in purposes:
            raise ValueError("Positive USD ceilings and named purposes required")
        contract = json.dumps({"limits": self.limits, "prices": PRICES, "price_date": PRICE_DATE}, sort_keys=True)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS contract (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS dispatch (id TEXT PRIMARY KEY, purpose TEXT NOT NULL, model TEXT NOT NULL, payload_sha TEXT NOT NULL, reserved INTEGER NOT NULL, charged INTEGER, status TEXT NOT NULL, response TEXT, request_id TEXT)")
            existing = db.execute("SELECT value FROM contract WHERE id=1").fetchone()
            if existing and existing[0] != contract:
                raise ValueError("Budget/price contract changed; do not reset a live spending ledger")
            db.execute("INSERT OR IGNORE INTO contract VALUES (1,?)", (contract,))

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def reserve(self, identity, purpose, model, payload):
        if purpose not in self.limits or purpose == "total" or model not in PRICES:
            raise ValueError("Unknown budget purpose or model price")
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        amount = request_reservation(model, payload)
        with self.transaction() as db:
            prior = db.execute("SELECT purpose,model,payload_sha,reserved,charged,status,response,request_id FROM dispatch WHERE id=?", (identity,)).fetchone()
            if prior:
                if prior[:3] != (purpose, model, fingerprint):
                    raise ValueError("A dispatch ID cannot be reused for a different request")
                return prior
            total = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM dispatch").fetchone()[0]
            used = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM dispatch WHERE purpose=?",(purpose,)).fetchone()[0]
            underestimated = db.execute("SELECT COUNT(*) FROM dispatch WHERE charged>reserved").fetchone()[0]
            if underestimated or total+amount > self.limits["total"] or used+amount > self.limits[purpose]:
                active_total = db.execute("SELECT COALESCE(SUM(reserved),0) FROM dispatch WHERE status='reserved'").fetchone()[0]
                active_used = db.execute("SELECT COALESCE(SUM(reserved),0) FROM dispatch WHERE status='reserved' AND purpose=?",(purpose,)).fetchone()[0]
                waitable = (not underestimated and total-active_total+amount<=self.limits['total']
                            and used-active_used+amount<=self.limits[purpose])
                raise BudgetExceeded("USD ceiling/reservations reached; no request was sent",waitable=waitable)
            db.execute("INSERT INTO dispatch VALUES (?,?,?,?,?,NULL,'reserved',NULL,NULL)", (identity,purpose,model,fingerprint,amount))
        return None

    def complete(self, identity, model, data, request_id):
        amount = usage_micro_usd(model, data["usage"], data.get("service_tier"))
        with self.transaction() as db:
            db.execute("UPDATE dispatch SET charged=?,status='completed',response=?,request_id=? WHERE id=?",
                       (amount,json.dumps(data,ensure_ascii=False),request_id,identity))

    def uncertain(self, identity):
        with self.transaction() as db:
            db.execute("UPDATE dispatch SET status='uncertain' WHERE id=? AND status='reserved'",(identity,))

    def snapshot(self):
        with self.transaction() as db:
            data = db.execute("SELECT purpose,COUNT(*),COALESCE(SUM(charged),0),SUM(CASE WHEN charged IS NULL THEN reserved ELSE 0 END) FROM dispatch GROUP BY purpose").fetchall()
        return {purpose: {"dispatches": count, "charged_usd_estimate": charged/1e6,
                          "unresolved_reservations_usd": held/1e6} for purpose,count,charged,held in data}


class BudgetTransport:
    def __init__(self, budget, purpose, transport=None):
        from .judges import request_json
        self.budget, self.purpose, self.transport = budget, purpose, transport or request_json

    def for_call(self, identity):
        parent = self
        class Bound:
            receipt = None
            def __call__(self, url, payload, key, timeout):
                if url != "https://api.openai.com/v1/responses":
                    raise ValueError("Budgeted transport permits only the official Responses endpoint")
                model = payload["model"]
                deadline=time.monotonic()+min(timeout,60)
                while True:
                    try:
                        prior = parent.budget.reserve(identity,parent.purpose,model,payload)
                        break
                    except BudgetExceeded as exc:
                        # A concurrent response can settle below its conservative
                        # reservation. Never release unknown/crashed dispatches.
                        if not exc.waitable or time.monotonic()>=deadline:
                            raise
                        time.sleep(.25)
                self.receipt = {"dispatch_id": identity, "replayed_response": prior is not None, "price_date": PRICE_DATE}
                if prior:
                    if prior[5] == "completed":
                        return json.loads(prior[6]), prior[7]
                    raise URLError("Earlier dispatch has uncertain billing; reservation retained, no redispatch")
                try:
                    data, request_id = parent.transport(url,payload,key,timeout)
                    parent.budget.complete(identity,model,data,request_id)
                    return data, request_id
                except BaseException:
                    parent.budget.uncertain(identity)
                    raise
        return Bound()
