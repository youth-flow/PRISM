"""Independent final-set evaluation for a staged full-pool LoRA PPO checkpoint.

This path is deliberately separate from the formal Empirical/PRISM packet.  It
may prepare a blind PPO-vs-SFT packet, but never dispatches a paid request.
"""
import csv
import io
from pathlib import Path

from prism.io import digest, rows
from .artifacts import Run, read_json, write_json


METHOD = "full-pool-lora"
SELECTION_STAGE = "lora-ppo-evaluation-selection"
COMPARISON = {"name": "full_pool_lora_vs_sft", "left": METHOD,
              "right": "sft", "primary": True}


def freeze_selection(c, episode):
    """Bind one already-probed PPO checkpoint before final-set generation."""
    if type(episode) is not int or episode not in c["ppo"]["checkpoints"]:
        raise ValueError("LoRA evaluation episode must be a configured PPO checkpoint")
    run = Run(c)
    if run.completed(SELECTION_STAGE):
        value = read_json(run.file("control/lora-ppo-evaluation-selection.json"))
        if value.get("episode") != episode:
            raise ValueError("Preserve the already frozen LoRA evaluation checkpoint")
        return value
    run.require("lora-ppo-admission")
    checkpoint = run.file(f"policy/{METHOD}/checkpoints/episode-{episode:08d}.pt")
    report = run.file(f"policy/{METHOD}/report-{episode:08d}.json")
    probes = [run.file(f"policy/{METHOD}/probes/{episode:08d}-{name}.json")
              for name in ("train_probe", "dev_probe")]
    latest_path = run.file(f"policy/{METHOD}/checkpoints/latest.json")
    latest = read_json(latest_path)
    report_value = read_json(report)
    world_size = latest.get("world_size")
    if (latest.get("episode") != episode or latest.get("path") != checkpoint.name
            or latest.get("sha256") != digest(checkpoint)
            or report_value.get("completed_episodes") != episode
            or world_size not in (4, 8)
            or report_value.get("distributed_world_size") != world_size):
        raise ValueError("LoRA evaluation checkpoint/report identity is not the selected staged stop")
    for name, path in zip(("train_probe", "dev_probe"), probes):
        value = read_json(path)
        if (value.get("episode") != episode or value.get("probe") != name
                or value.get("checkpoint_sha256") != digest(checkpoint)
                or value.get("shared_probe_sha256") is None):
            raise ValueError("LoRA evaluation probe does not bind the selected checkpoint")
    latest_snapshot = run.file(f"control/lora-ppo-evaluation-latest-{episode:08d}.json")
    write_json(latest_snapshot, latest)
    value = {
        "protocol": "v3_full_pool_lora_blind_evaluation_selection_v2",
        "method": METHOD,
        "episode": episode,
        "checkpoint": f"policy/{METHOD}/checkpoints/{checkpoint.name}",
        "checkpoint_sha256": digest(checkpoint),
        "report_sha256": digest(report),
        "probe_sha256": {path.stem.rsplit("-", 1)[-1]: digest(path) for path in probes},
        "latest_snapshot": f"control/{latest_snapshot.name}",
        "latest_snapshot_sha256": digest(latest_snapshot),
        "contract_hash": latest["contract_hash"],
        "distributed_world_size": world_size,
        "authority": "user_continued_independent_evaluation_after_32k_review",
        "claim_scope": "prepare independent blind PPO-vs-SFT evaluation; no paid calls",
    }
    target = run.file("control/lora-ppo-evaluation-selection.json")
    write_json(target, value)
    run.publish(SELECTION_STAGE, [target, latest_snapshot],
                [run.receipt_path("lora-ppo-admission"), checkpoint, report,
                 *probes],
                {"episode": episode, "checkpoint_sha256": value["checkpoint_sha256"]})
    return value


def _selection(c):
    run = Run(c)
    run.require(SELECTION_STAGE)
    value = read_json(run.file("control/lora-ppo-evaluation-selection.json"))
    checkpoint = run.file(value["checkpoint"])
    latest_snapshot = run.file(value.get("latest_snapshot", "__invalid__"))
    latest = read_json(latest_snapshot)
    if (value.get("protocol") != "v3_full_pool_lora_blind_evaluation_selection_v2"
            or value.get("method") != METHOD or digest(checkpoint) != value.get("checkpoint_sha256")):
        raise ValueError("Invalid frozen LoRA evaluation selection")
    if (digest(latest_snapshot) != value.get("latest_snapshot_sha256")
            or latest.get("episode") != value.get("episode")
            or latest.get("path") != checkpoint.name
            or latest.get("sha256") != value.get("checkpoint_sha256")
            or latest.get("contract_hash") != value.get("contract_hash")
            or latest.get("world_size") != value.get("distributed_world_size")):
        raise ValueError("Invalid frozen LoRA evaluation latest-pointer snapshot")
    return run, value, checkpoint


def load_selected_actor(c):
    """Rebuild the SFT+LoRA actor and restore only the selected actor tensors."""
    import torch
    from .distributed_ppo import PROTOCOL
    from .ppo import load_model, load_tokenizer, make_actor
    from .ppo_trainer import _restore_trainable, seed_everything

    _, selection, checkpoint = _selection(c)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (state.get("protocol") != PROTOCOL or state.get("episode") != selection["episode"]
            or state.get("contract_hash") != selection["contract_hash"]
            or state.get("world_size") != selection["distributed_world_size"]
            or not isinstance(state.get("actor"), dict) or not state["actor"]):
        raise ValueError("Selected LoRA evaluation checkpoint state failed its training contract")
    seed_everything(c["study"]["seed"])
    actor = make_actor(load_model(c), c["ppo"])
    _restore_trainable(actor, state["actor"])
    actor.eval()
    return actor, load_tokenizer(c)


def _sources(c, run):
    from .evaluation import _prompts, _validate_row
    prompts = _prompts(c, run)
    sources = {}
    for method in ("sft", METHOD):
        run.require(f"eval-{method}")
        records = rows(run.file(f"evaluation/{method}/responses.jsonl"))
        if len(records) != len(prompts):
            raise ValueError("LoRA blind evaluation answer coverage differs")
        for index, (record, prompt) in enumerate(zip(records, prompts)):
            _validate_row(c, record, prompt, index, method)
        sources[method] = records
    return prompts, sources


def export_judge(c):
    """Create an immutable 500-prompt AB/BA packet without making API calls."""
    from prism.judges import APIJudge, evaluation_call_seed, judge_protocol
    from prism.local_judge import _hash
    from .evaluation import JUDGE_KEYS

    run, selection, _ = _selection(c)
    if run.completed("lora-ppo-judge-export"):
        return run.require("lora-ppo-judge-export")
    prompts, sources = _sources(c, run)
    settings = {key: c["evaluation"][key] for key in JUDGE_KEYS}
    dependencies = [run.receipt_path(SELECTION_STAGE), run.receipt_path("data"),
                    run.receipt_path("eval-sft"), run.receipt_path(f"eval-{METHOD}")]
    tasks = []
    for index, prompt in enumerate(prompts):
        for oi, order in enumerate(c["evaluation"]["answer_orders"]):
            seed_config = {"evaluation": {"seed": c["evaluation"]["seed"]}}
            calls = [{"seed": evaluation_call_seed(seed_config, selection["episode"],
                                                    c["study"]["seed"], 0, index, oi, attempt),
                      "flip": order == "BA"}
                     for attempt in range(settings["max_attempts"])]
            tasks.append({"id": f"{COMPARISON['name']}:{index}:{order}",
                          "messages": prompt["messages"],
                          "left": sources[METHOD][index]["response"],
                          "right": sources["sft"][index]["response"], "calls": calls})
    packet = {
        "kind": "prism_local_judge", "version": 1, "stage": "lora-ppo-evaluate-judge",
        "method": "paired-v3-lora", "role": "evaluation",
        "budget": selection["episode"], "seed": c["study"]["seed"],
        "judge_protocol": judge_protocol("evaluation", "openai"), "settings": settings,
        "request_endpoint": APIJudge({"judges": {"evaluation": settings}}, "evaluation").url,
        "config_sha256": run.fingerprint,
        "input_sha256": {run._key(path): digest(path) for path in dependencies},
        "tasks": tasks, "prior_calls": [],
        "v3": {"protocol": "hh_finite_pool_v3", "path": "full_pool_lora_ppo_blind_v1",
               "prompts": len(prompts), "comparisons": [COMPARISON],
               "valid_judgments": len(tasks), "policy_episode": selection["episode"]},
    }
    packet["packet_id"] = _hash(packet)
    target = run.file("judge/lora-ppo/packet.json")
    write_json(target, packet)
    return run.publish("lora-ppo-judge-export", [target], dependencies,
                       {"tasks": len(tasks), "packet_id": packet["packet_id"],
                        "episode": selection["episode"]})


def _bound_packet(c, run):
    from prism.local_judge import _load_packet
    run.require("lora-ppo-judge-export")
    packet = _load_packet(run.file("judge/lora-ppo/packet.json"))
    if (packet.get("v3", {}).get("protocol") != "hh_finite_pool_v3"
            or packet.get("v3", {}).get("path") != "full_pool_lora_ppo_blind_v1"
            or packet.get("method") != "paired-v3-lora" or packet["config_sha256"] != run.fingerprint):
        raise ValueError("Judge packet is not the frozen LoRA PPO blind comparison")
    for key, expected in packet["input_sha256"].items():
        if digest(run._resolve(key)) != expected:
            raise ValueError("LoRA PPO Judge packet input changed")
    return packet


def import_judge(c, results):
    """Import an explicitly paid local result into the isolated LoRA path."""
    from prism.local_judge import _records
    from .evaluation import _billing_path, _validate_billing
    run = Run(c)
    packet = _bound_packet(c, run)
    _, interpreted, accounting = _records(packet, results, with_accounting=True)
    complete = {}
    for task in packet["tasks"]:
        successful = [interpreted[(task["id"], attempt)]["label"]
                      for attempt in range(len(task["calls"]))
                      if (task["id"], attempt) in interpreted
                      and interpreted[(task["id"], attempt)]["label"] is not None]
        if len(successful) != 1:
            raise ValueError("LoRA PPO evaluation requires one valid result per AB/BA task")
        complete[task["id"]] = successful[0]
    results = Path(results)
    billing_path = Path(str(results) + ".billing.json")
    money = _validate_billing(packet, results, read_json(billing_path))
    target = run.file("judge/lora-ppo/results.import.json")
    value = {"packet_id": packet["packet_id"], "labels": complete,
             "accounting": accounting, "billing": money}
    write_json(target, value)
    return run.publish("lora-ppo-judge-import", [target],
                       [run.receipt_path("lora-ppo-judge-export"), results, billing_path],
                       {"valid_judgments": len(complete), **accounting, **money})


def aggregate(c):
    """Aggregate only the isolated LoRA PPO-vs-SFT blind comparison."""
    from .evaluation import _generation_metrics, score_comparison
    run, selection, _ = _selection(c)
    if run.completed("lora-ppo-aggregate"):
        return run.require("lora-ppo-aggregate")
    run.require("lora-ppo-judge-import")
    packet = _bound_packet(c, run)
    imported = read_json(run.file("judge/lora-ppo/results.import.json"))
    if imported["packet_id"] != packet["packet_id"]:
        raise ValueError("Imported labels belong to another LoRA PPO packet")
    prompts, sources = _sources(c, run)
    comparison = score_comparison(c, COMPARISON, prompts, imported["labels"],
                                  bootstrap_seed=c["evaluation"]["seed"] + 32768)
    value = {
        "protocol": "hh_finite_pool_lora_ppo_blind_v1", "selection": selection,
        "seed": c["study"]["seed"], "judge": packet["settings"],
        "comparison": comparison,
        "generation": {method: _generation_metrics(records) for method, records in sources.items()},
        "accounting": imported["accounting"], "billing": imported["billing"],
        "uncertainty_scope": "Prompt bootstrap conditional on one training seed",
    }
    target = run.file("summaries/lora-ppo-vs-sft.json")
    write_json(target, value)
    fields = ["name", "left", "right", "primary", "prompts", "win_rate",
              "standard_error", "ci95", "position_disagreement_rate"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerow({key: comparison.get(key) for key in fields})
    csv_path = run.file("summaries/lora-ppo-vs-sft.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    text = stream.getvalue().replace("\r\n", "\n")
    if csv_path.exists() and csv_path.read_text(encoding="utf-8") != text:
        raise FileExistsError("Preserve different LoRA PPO summary CSV")
    if not csv_path.exists():
        csv_path.write_text(text, encoding="utf-8")
    return run.publish("lora-ppo-aggregate", [target, csv_path],
                       [run.receipt_path("lora-ppo-judge-import")],
                       {"win_rate": comparison["win_rate"], "prompts": len(prompts)})
