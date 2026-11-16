"""Local shared USD reservations and recoverable API dispatches.

Unknown/failed dispatches retain their full reservation. A completed response is
saved before the caller's JSONL append, so replay after a crash never resends it.
Prices are an explicit snapshot, not a substitute for the provider invoice.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Lock, RLock
import time
from urllib.error import URLError


# USD / million tokens: ordinary, cache read, 30m write, 12h write, output.
PRICES = {"gpt-6.1-sol": (2, .1, 2.5, 4, 10),
          "gpt-6-luna": (.1, .01, .125, .2, .5)}
PRICE_DATE = "2026-11-16"
_LOCKS = {}
_LOCKS_GUARD = Lock()


def _path_lock(path):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(os.path.normcase(str(path.resolve())), RLock())


def journal_mode(version):
    # SQLite's published WAL-reset fix: 3.51.3+, or its 3.44.6/3.50.7
    # backports. Keep older bundled Python runtimes on rollback journals.
    safe=version>=(3,51,3) or (version[:2]==(3,50) and version>=(3,50,7)) or (version[:2]==(3,44) and version>=(3,44,6))
    return 'WAL' if safe else 'DELETE'


class DispatchPaused(RuntimeError):
    pass


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
        self._lock = _path_lock(self.path)
        self.response_dir = self.path.with_name(self.path.name+'.responses')
        # Serialize same-process writes; use WAL only on fixed SQLite releases.
        # No package upgrade or WAL on an unpatched bundled runtime is needed.
        with self._lock:
            db=sqlite3.connect(self.path,timeout=60)
            try:
                db.execute('PRAGMA journal_mode='+journal_mode(sqlite3.sqlite_version_info))
            finally:
                db.close()
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

    def tighten_limits(self, *, total_usd, purposes, reason):
        """Lower live ceilings without resetting responses or dispatch identity.

        SQLite triggers also constrain workers holding the old in-memory limits.
        Existing/unknown reservations remain charged against the new ceilings.
        """
        return self._change_limits(total_usd,purposes,reason,reallocate=False)

    def reallocate_limits(self, *, purposes, reason):
        """Move purpose allocations after explicit approval; total cannot rise."""
        with self.transaction() as db:
            current=self._effective_limits(db)
        return self._change_limits(current['total']/1e6,purposes,reason,reallocate=True)

    def _effective_limits(self, db):
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='live_ceiling'").fetchone():
            return dict(db.execute("SELECT purpose,amount FROM live_ceiling").fetchall()) or self.limits
        return self.limits

    def _change_limits(self,total_usd,purposes,reason,*,reallocate):
        requested={"total":micro_usd(total_usd),**{k:micro_usd(v) for k,v in purposes.items()}}
        if set(requested)!=set(self.limits) or any(v<=0 for v in requested.values()) or not reason.strip():
            raise ValueError("Specify every existing positive ceiling and an audit reason")
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS live_ceiling (purpose TEXT PRIMARY KEY, amount INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS ceiling_change (id INTEGER PRIMARY KEY, changed_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, previous TEXT NOT NULL, revised TEXT NOT NULL, reason TEXT NOT NULL)")
            current=self._effective_limits(db)
            if (requested['total']>current['total'] or any(requested[k]>self.limits[k] for k in requested)
                    or (not reallocate and any(requested[k]>current[k] for k in requested))):
                raise ValueError("Live ceilings can only tighten; never reset spent amounts")
            usage={k:0 for k in requested}
            for purpose,amount in db.execute("SELECT purpose,SUM(COALESCE(charged,reserved)) FROM dispatch GROUP BY purpose"):
                usage[purpose]=amount;usage['total']+=amount
            if any(usage[k]>requested[k] for k in requested):
                raise BudgetExceeded("New ceiling is below already charged/reserved requests; no changes applied")
            for purpose,amount in requested.items():
                db.execute("INSERT OR REPLACE INTO live_ceiling VALUES (?,?)",(purpose,amount))
            db.execute("INSERT INTO ceiling_change (previous,revised,reason) VALUES (?,?,?)",
                       (json.dumps(current,sort_keys=True),json.dumps(requested,sort_keys=True),reason))
            db.execute("""CREATE TRIGGER IF NOT EXISTS enforce_live_usd_ceiling
                BEFORE INSERT ON dispatch BEGIN
                SELECT CASE WHEN
                    COALESCE((SELECT SUM(COALESCE(charged,reserved)) FROM dispatch),0)+NEW.reserved
                        >(SELECT amount FROM live_ceiling WHERE purpose='total')
                    OR COALESCE((SELECT SUM(COALESCE(charged,reserved)) FROM dispatch WHERE purpose=NEW.purpose),0)+NEW.reserved
                        >(SELECT amount FROM live_ceiling WHERE purpose=NEW.purpose)
                    THEN RAISE(ABORT,'Live USD ceiling reached; no request was sent') END;
                END""")
        return {k:v/1e6 for k,v in requested.items()}

    @contextmanager
    def transaction(self):
        with self._lock:
            db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            try:
                db.execute('PRAGMA synchronous=FULL')
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
            finally:
                db.close()

    @classmethod
    def open_existing(cls,path):
        """Use the original immutable contract, never reset old spending."""
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError('Existing USD ledger required')
        db=sqlite3.connect(path,timeout=60)
        try:
            contract=json.loads(db.execute('SELECT value FROM contract WHERE id=1').fetchone()[0])
        finally:
            db.close()
        if contract['prices']!={k:list(v) for k,v in PRICES.items()} or contract['price_date']!=PRICE_DATE:
            raise ValueError('Ledger price snapshot changed')
        limits=contract['limits']
        return cls(path,total_usd=limits['total']/1e6,purposes={k:v/1e6 for k,v in limits.items() if k!='total'})

    def control(self,paused,reason):
        if type(paused) is not bool or not reason.strip():
            raise ValueError('An explicit pause state and reason are required')
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS dispatch_control (id INTEGER PRIMARY KEY, paused INTEGER NOT NULL, reason TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS admission_events (id INTEGER PRIMARY KEY, timestamp_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, action TEXT NOT NULL, reason TEXT NOT NULL)')
            db.execute('INSERT OR REPLACE INTO dispatch_control VALUES (1,?,?)',(int(paused),reason))
            db.execute('INSERT INTO admission_events (action,reason) VALUES (?,?)',('pause' if paused else 'resume',reason))
            db.execute("""CREATE TRIGGER IF NOT EXISTS drain_api_dispatches BEFORE INSERT ON dispatch
                WHEN (SELECT paused FROM dispatch_control WHERE id=1)=1 BEGIN
                SELECT RAISE(ABORT,'API dispatch paused; no HTTP request was sent'); END""")

    def state(self):
        with self.transaction() as db:
            limits={k:v/1e6 for k,v in self._effective_limits(db).items()}
            mode=db.execute('PRAGMA journal_mode').fetchone()[0]
            control=None
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dispatch_control'").fetchone():
                control=db.execute('SELECT paused,reason FROM dispatch_control WHERE id=1').fetchone()
        return {'limits_usd':limits,'paused':bool(control and control[0]),'pause_reason':control[1] if control else None,
                'sqlite_version':sqlite3.sqlite_version,'journal_mode':mode,'usage':self.snapshot()}

    def cache_response(self,identity,model,payload,data,request_id):
        """Persist a received HTTP response before contended SQL settlement."""
        self.response_dir.mkdir(parents=True,exist_ok=True)
        value={'identity':identity,'model':model,'payload_sha':hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
               'response':data,'request_id':request_id}
        target=self.response_dir/(hashlib.sha256(identity.encode()).hexdigest()+'.json')
        temporary=None
        try:
            with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=self.response_dir,delete=False) as stream:
                temporary=Path(stream.name)
                json.dump(value,stream,ensure_ascii=False,allow_nan=False)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,target)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()

    def cached_response(self,identity,model,payload):
        target=self.response_dir/(hashlib.sha256(identity.encode()).hexdigest()+'.json')
        if not target.exists():return None
        value=json.loads(target.read_text(encoding='utf-8'))
        fingerprint=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if (value['identity'],value['model'],value['payload_sha'])!=(identity,model,fingerprint):
            raise ValueError('Cached HTTP response identity changed')
        return value['response'],value['request_id']

    def reserve(self, identity, purpose, model, payload):
        if purpose not in self.limits or purpose == "total" or model not in PRICES:
            raise ValueError("Unknown budget purpose or model price")
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        amount = request_reservation(model, payload)
        with self.transaction() as db:
            limits=self._effective_limits(db)
            prior = db.execute("SELECT purpose,model,payload_sha,reserved,charged,status,response,request_id FROM dispatch WHERE id=?", (identity,)).fetchone()
            if prior:
                if prior[:3] != (purpose, model, fingerprint):
                    raise ValueError("A dispatch ID cannot be reused for a different request")
                return prior
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dispatch_control'").fetchone():
                control=db.execute('SELECT paused FROM dispatch_control WHERE id=1').fetchone()
                if control and control[0]:raise DispatchPaused('API dispatch is paused; no request was sent')
            total = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM dispatch").fetchone()[0]
            used = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM dispatch WHERE purpose=?",(purpose,)).fetchone()[0]
            underestimated = db.execute("SELECT COUNT(*) FROM dispatch WHERE charged>reserved").fetchone()[0]
            if underestimated or total+amount > limits["total"] or used+amount > limits[purpose]:
                active_total = db.execute("SELECT COALESCE(SUM(reserved),0) FROM dispatch WHERE status='reserved'").fetchone()[0]
                active_used = db.execute("SELECT COALESCE(SUM(reserved),0) FROM dispatch WHERE status='reserved' AND purpose=?",(purpose,)).fetchone()[0]
                waitable = (not underestimated and total-active_total+amount<=limits['total']
                            and used-active_used+amount<=limits[purpose])
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
                deadline=time.monotonic()+min(timeout,900)
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
                    cached=parent.budget.cached_response(identity,model,payload)
                    if cached is not None:
                        data,request_id=cached
                        parent.budget.complete(identity,model,data,request_id)
                        return data,request_id
                    raise URLError("Earlier dispatch has uncertain billing; reservation retained, no redispatch")
                try:
                    data, request_id = parent.transport(url,payload,key,timeout)
                except BaseException:
                    try:parent.budget.uncertain(identity)
                    except sqlite3.OperationalError:pass  # Retain reserved amount; preserve the original failure.
                    raise
                parent.budget.cache_response(identity,model,payload,data,request_id)
                parent.budget.complete(identity,model,data,request_id)
                return data, request_id
        return Bound()
