"""Explicit standard-processing/deadline authorization, separate from study config."""
import hashlib,json
from pathlib import Path
import yaml

NAME='execution.v2-deadline.yaml'


def canonical_path():
    return Path(__file__).resolve().parents[2]/'configs'/NAME


def authorization(path=None):
    path=Path(path) if path is not None else canonical_path()
    if path.read_bytes()!=canonical_path().read_bytes():
        raise ValueError('Deadline execution authorization changed')
    value=yaml.safe_load(path.read_text(encoding='utf-8'))
    if (value.get('kind')!='prism_v2_deadline_execution' or value.get('version')!=1
            or value.get('authorization')!='user_sets_evaluation_cap_30_and_requests_completion_before_0700'
            or value.get('active_budgets')!=[2048,4096] or value.get('cancelled_budgets')!=[8192]
            or value.get('transport_overrides')!={'stream':True,'allowed_service_tiers':['default','flex']}
            or value.get('spending_usd')!={'operational_total':63,'calibration':2,'training':31,'evaluation':30}):
        raise ValueError('Invalid deadline execution authorization')
    return value


def authorization_sha256():
    authorization()
    return hashlib.sha256(canonical_path().read_bytes()).hexdigest()


def wire_evidence(payload, *, standard=False):
    wire={**payload,'stream':True}
    if standard:
        if payload.get('service_tier')!='flex':raise ValueError('Standard override requires the original Flex request')
        wire['service_tier']='default'
    evidence={'protocol':'responses_sse_standard_v1' if standard else 'responses_sse_v1',
              'effective_payload_sha256':hashlib.sha256(json.dumps(wire,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
    if standard:evidence['execution_authorization_sha256']=authorization_sha256()
    return wire,evidence
