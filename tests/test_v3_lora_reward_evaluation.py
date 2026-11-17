import numpy as np
import pytest

from prism.io import digest
from prism.v3.artifacts import write_json
from prism.v3.lora_reward import CACHE_PREFIX, PREFIX, STAGE_PREFIX
from prism.v3.lora_reward_evaluation import (checkpoint_targets, independent_metrics,
                                             summarize, validate_checkpoint)
from v3_helpers import configuration, numerical_run


def test_independent_metrics_orientation_and_no_unseen_training_accuracy():
    m = independent_metrics([[2, 0], [0, 2], [0, 0]], [1, 0, 1])
    assert m["accuracy"] == pytest.approx(2.5/3)
    assert m["bt_nll"] == pytest.approx((2*np.logaddexp(0, -2)+np.log(2))/3)
    assert independent_metrics(np.empty((0, 2)), []) is None
    with pytest.raises(ValueError):
        independent_metrics([[float('nan'), 0]], [1])


def test_true_optimizer_state_validation():
    torch = pytest.importorskip("torch")
    target, contract = {"step": 2, "pairs_seen": 16}, {"test": True}
    state = {"trainable": {"head": torch.ones(2)}, "step": 2, "seen": 16, "contract": contract,
             "optimizer": {"state": {0: {"step": torch.tensor(2.), "exp_avg": torch.zeros(2)}}}}
    validate_checkpoint(state, target, contract)
    state["optimizer"]["state"][0]["step"] = torch.tensor(1.)
    with pytest.raises(ValueError, match="Adam step"):
        validate_checkpoint(state, target, contract)


def test_synthetic_checkpoint_shards_merge_only_consumed_training_prefix(tmp_path):
    c = configuration(tmp_path)
    run = numerical_run(c)
    from prism.io import rows
    datasets = {"train": rows(run.file("data/pool.jsonl")), "dev": rows(run.file("data/reward_dev.jsonl"))}
    for split, records in datasets.items():
        path = run.file(f"{CACHE_PREFIX}/{split}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.tile(np.array([1, 4, 2, 1, 5, 2], dtype=np.int32), len(records)).tofile(path.with_suffix(".tokens"))
        offsets = np.array([[i*6, i*6+3, i*6+6] for i in range(len(records))])
        labels = np.array([int(i < (3 if split == "train" else 2)) for i in range(len(records))])
        np.savez(path.with_suffix(".npz"), offsets=offsets, ids=np.array([r['id'] for r in records]), labels=labels)
    initial = independent_metrics(np.zeros((4, 2)), [1, 1, 0, 0])
    full_scores = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    full = independent_metrics(full_scores, [1, 1, 0, 0])
    training = {"training_pairs": 6, "best_step": 0,
                "history": [{"step": 0, "pairs_seen": 0, "development": initial},
                            {"step": 1, "pairs_seen": 6, "development": full}],
                "milestone_checkpoints": {"6": f"{PREFIX}/checkpoints/step-000001.pt"},
                "selected_checkpoint": f"{PREFIX}/checkpoints/step-000000.pt"}
    targets = checkpoint_targets(training)
    weights = []
    for step in range(2):
        path = run.file(f"{PREFIX}/checkpoints/step-{step:06d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic provenance bytes "+str(step).encode())
        weights.append(path)
    marker = run.file(f"{PREFIX}/training.json")
    write_json(marker, training)
    run.publish(f"{STAGE_PREFIX}-train", [marker, *weights], [run.receipt_path("data")])
    order = np.random.default_rng(11).permutation(6)
    for shard in range(2):
        outputs, snapshots = [], {}
        for step, seen in [(0, 0), (1, 6)]:
            ii, jj = order[:seen][shard::2], np.arange(shard, 4, 2)
            path = run.file(f"{PREFIX}/evaluation/step-{step:06d}-part-{shard:03d}-of-002.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, training_indices=ii, training_ids=np.array([datasets['train'][i]['id'] for i in ii]),
                     training_scores=np.zeros((len(ii), 2)), development_indices=jj,
                     development_ids=np.array([datasets['dev'][i]['id'] for i in jj]),
                     development_scores=np.zeros((len(jj), 2)) if step == 0 else full_scores[jj])
            snapshots[str(step)] = {"path": str(path.relative_to(run.path)), "pairs_seen": seen,
                                    "checkpoint_sha256": digest(weights[step]), "optimizer_state_passed": True}
            outputs.append(path)
        marker = run.file(f"{PREFIX}/evaluation/part-{shard:03d}-of-002.json")
        write_json(marker, {"shard_index": shard, "shards": 2, "targets": targets, "snapshots": snapshots})
        run.publish(f"{STAGE_PREFIX}-score-{shard:03d}-of-002", [*outputs, marker], [run.receipt_path(f"{STAGE_PREFIX}-train")])
    baseline = run.file("diagnostics/full-pool-reward/l2-0.01/report.json")
    write_json(baseline, {"training": {"accuracy": .6}, "development": {"accuracy": .6}})
    result = summarize(c, 2)
    assert result["selected"]["training"] is None
    assert result["selected"]["pairs_seen"] == 0
    assert result["learning_curve"][0]["training"]["pairs"] == 6
    assert result["learning_curve"][0]["development"]["accuracy"] == 1.
    assert result["independent_shard_coverage_passed"]
    assert summarize(c, 2) == result
