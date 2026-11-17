"""Crash after output creation must not strand immutable stage paths."""
import json

import pytest

from prism.io import digest
from prism.v3.artifacts import Run, read_json
from prism.v3.data import encode_pair, prepare_data
from prism.v3.features import extract_features, merge_features
from prism.v3.geometry import build_geometry, design_report, solve_design
from test_v3_data import FakeTokenizer, tiny_config, write_sources
from test_v3_features import deterministic_scorer


def fail_publication(monkeypatch, stage):
    original = Run.publish
    def interrupted(self, name, *args, **kwargs):
        if name == stage:
            raise OSError("injected receipt publication interruption")
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(Run, "publish", interrupted)
    return original


def prepared(c):
    prepare_data(c, write_sources(c), tokenizer=FakeTokenizer())


def test_feature_publication_recovery_does_not_repeat_any_model_forward(tmp_path, monkeypatch):
    c = tiny_config(tmp_path)
    prepared(c)
    class CountingScorer:
        def __init__(self):
            self.calls = 0
        def __call__(self, row):
            self.calls += 1
            return deterministic_scorer(row)
    scorer = CountingScorer()
    original = fail_publication(monkeypatch, "features-part-00000-of-00001")
    with pytest.raises(OSError, match="interruption"):
        extract_features(c, scorer=scorer)
    completed_forwards = scorer.calls
    run = Run(c)
    output = run.file("features/part-00000-of-00001.npz")
    original_hash = digest(output)
    monkeypatch.setattr(Run, "publish", original)
    report = extract_features(c, scorer=scorer)
    assert scorer.calls == completed_forwards
    assert digest(output) == original_hash
    assert report["pool_rows"] == 18
    run.require("features-part-00000-of-00001")


def test_untrusted_feature_attempt_is_archived_with_hashes_before_recompute(tmp_path, monkeypatch):
    c = tiny_config(tmp_path)
    prepared(c)
    original = fail_publication(monkeypatch, "features-part-00000-of-00001")
    with pytest.raises(OSError):
        extract_features(c, scorer=deterministic_scorer)
    run = Run(c)
    output = run.file("features/part-00000-of-00001.npz")
    output.write_bytes(output.read_bytes() + b"partial/mutated attempt")
    failed_hash = digest(output)
    monkeypatch.setattr(Run, "publish", original)
    extract_features(c, scorer=deterministic_scorer)
    manifests = list(run.file("attempts/features-part-00000-of-00001").glob("*/manifest.json"))
    assert len(manifests) == 1
    preserved = read_json(manifests[0])
    npz_entry = next(entry for entry in preserved["files"] if entry["original"] == "features/part-00000-of-00001.npz")
    assert npz_entry["sha256"] == failed_hash
    assert digest(run.file(npz_entry["archived"])) == failed_hash
    assert digest(output) != failed_hash
    run.require("features-part-00000-of-00001")


def test_data_receipt_recovery_uses_sealed_outputs_without_reencoding(tmp_path, monkeypatch):
    c = tiny_config(tmp_path)
    marker = write_sources(c)
    class CountingTokenizer(FakeTokenizer):
        def __init__(self):
            self.calls = 0
        def encode(self, *args, **kwargs):
            self.calls += 1
            return super().encode(*args, **kwargs)
    tokenizer = CountingTokenizer()
    original = fail_publication(monkeypatch, "data")
    with pytest.raises(OSError):
        prepare_data(c, marker, tokenizer=tokenizer)
    before = tokenizer.calls
    monkeypatch.setattr(Run, "publish", original)
    result = prepare_data(c, marker, tokenizer=tokenizer)
    assert tokenizer.calls == before and result["valid_pool_pairs"] == 18
    Run(c).require("data")


@pytest.mark.parametrize("stage", ["features", "geometry", "design-0.01", "design-report"])
def test_later_stage_sealed_publication_can_finish_after_interruption(tmp_path, monkeypatch, stage):
    c = tiny_config(tmp_path)
    prepared(c)
    extract_features(c, scorer=deterministic_scorer)
    operations = [("features", lambda: merge_features(c, 1)), ("geometry", lambda: build_geometry(c)),
                  ("design-0.001", lambda: solve_design(c, .001)),
                  ("design-0.01", lambda: solve_design(c, .01)),
                  ("design-0.1", lambda: solve_design(c, .1)),
                  ("design-report", lambda: design_report(c))]
    for name, operation in operations:
        if name == stage:
            original = fail_publication(monkeypatch, stage)
            with pytest.raises(OSError):
                operation()
            monkeypatch.setattr(Run, "publish", original)
            recovered = operation()
            assert isinstance(recovered, dict)
            Run(c).require(stage)
            break
        operation()


def test_reserved_padding_is_never_accepted_as_human_text(tmp_path):
    c = tiny_config(tmp_path)
    from prism.v3.data import canonical_pair
    from test_v3_data import raw_pair
    row, _ = canonical_pair(raw_pair("q"), {})
    class PaddingTokenizer(FakeTokenizer):
        pad_token_id = 5
        def encode(self, text, add_special_tokens=False):
            return [5]
    with pytest.raises(ValueError, match="reserved_padding"):
        encode_pair(row, PaddingTokenizer(), c["data"]["context_tokens"])
