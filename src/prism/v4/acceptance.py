"""Explicit engineering acceptance, not formal Reward/PPO training."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from prism.io import digest, rows
from prism.v3.artifacts import Run, read_json, write_json
from .config import require_runtime


def require_policy_admission(c):
    run = Run(c, create=False)
    path = run.file("control/policy-admission.json")
    if not path.is_file():
        raise ValueError("V4 formal PPO is paused; review Reward/Design and real policy-smoke before admission")
    value = read_json(path)
    stages = ("cohort-audit", "model-smoke", "policy-smoke", "study-selection", "design-report",
              "reward", "training-audit", "normalization")
    expected = {}
    for stage in stages:
        run.require(stage)
        expected[stage] = digest(run.receipt_path(stage))
    if (value.get("authority") != "explicit_user_after_v4_reward_review" or value.get("user_authorized") is not True
            or value.get("config_sha256") != run.fingerprint or value.get("receipts") != expected):
        raise ValueError("V4 PPO admission must bind the reviewed config, Reward, Design and GPU acceptance")
    if not read_json(run.file("diagnostics/policy-smoke/report.json"))["passed"]:
        raise ValueError("V4 real policy smoke has not passed")
    return value


def prepare_reference_data(c, reference_run, *, tokenizer=None):
    """Re-tokenize locked human cohorts; never filter or resample V3 identities."""
    from prism.io import write_rows
    from prism.hf import prompt_ids
    from prism.v3.data import encode_pair, tokenizer_fingerprint
    from prism.v3.ppo import load_tokenizer
    from prism.v3.recovery import stage_lock, recover_stage, seal_and_publish
    if not reference_run:
        raise ValueError("V4 prepare-data requires --reference-run: preserve the V3 human-data comparison")
    reference = Path(reference_run).resolve()
    run = Run(c)
    names = ("pool", "labels", "policy", "ppo_dev", "evaluation", "reward_dev", "rejected", "provenance")
    reference_marker = read_json(reference / "experiment.json")
    if reference_marker["protocol"] != "hh_finite_pool_v3":
        raise ValueError("Reference must be the V3 HH run")
    receipt = read_json(reference / "receipts/data.json")
    if receipt["config_sha256"] != reference_marker["config_sha256"]:
        raise ValueError("Reference data/config binding changed")
    dependencies = [reference / "experiment.json", reference / "receipts/data.json"]
    for key, expected in receipt["outputs"].items():
        if not key.startswith("run:data/"):
            raise ValueError("Unexpected reference data output path")
        path = (reference / key[4:]).resolve()
        if not path.is_relative_to(reference) or digest(path) != expected:
            raise ValueError("Reference data output/hash mismatch")
        dependencies.append(path)
    outputs = [run.file(f"data/{name}.jsonl") for name in names] + [run.file("data/manifest.json")]
    with stage_lock(run, "data"):
        if run.completed("data"):
            return read_json(run.file("data/manifest.json"))
        context = {"reference_config": reference_marker["config_sha256"], "model": c["model"]}
        sources = [Path(__file__), Path(__file__).parent / "models.py"]
        if recover_stage(run, "data", outputs, dependencies, code_paths=sources, context=context):
            return read_json(run.file("data/manifest.json"))
        tokenizer = load_tokenizer(c) if tokenizer is None else tokenizer
        records = {name: rows(reference / f"data/{name}.jsonl") for name in names}
        maximum = 0
        for name in ("pool", "reward_dev"):
            for r in records[name]:
                prompt, responses = encode_pair(r, tokenizer, c["data"]["context_tokens"])
                r["prompt_tokens"] = len(prompt)
                maximum = max(maximum, *(len(prompt) + len(response) for response in responses))
        for name in ("policy", "ppo_dev", "evaluation"):
            for r in records[name]:
                prompt_ids(tokenizer, r["messages"], c["data"]["max_prompt_tokens"])
        report = read_json(reference / "data/manifest.json")
        report.update(protocol="hh_v4_locked_cohorts_retokenized_v1", tokenizer=c["model"],
                      tokenizer_semantics=tokenizer_fingerprint(tokenizer), context_tokens=c["data"]["context_tokens"],
                      reference_config_sha256=reference_marker["config_sha256"],
                      canonical_cohorts_preserved=True, token_ids_reused=False, maximum_pair_tokens=maximum)
        for name in names:
            write_rows(run.file(f"data/{name}.jsonl"), records[name])
        write_json(outputs[-1], report)
        seal_and_publish(run, "data", outputs, dependencies, code_paths=sources, context=context, metadata=report)
        return report


def runtime_check(c):
    import torch
    from transformers import AutoConfig, Olmo3Config, Olmo3ForCausalLM
    return {"passed": True, "packages": require_runtime(c), "torch": torch.__version__,
            "olmo3_registered": AutoConfig.for_model("olmo3").model_type == "olmo3",
            "model_downloaded": False, "gpu_used": False}


def download_model(c):
    from huggingface_hub import snapshot_download
    require_runtime(c)
    path = Path(snapshot_download(repo_id=c["model"]["id"], revision=c["model"]["revision"],
                                 allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer.model"]))
    if path.name != c["model"]["revision"]:
        raise ValueError("Downloaded snapshot does not match the pinned revision")
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if cfg.model_type != "olmo3" or cfg.hidden_size != 4096:
        raise ValueError("Wrong downloaded architecture")
    files = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in path.iterdir() if p.is_file()}
    report = {"model": c["model"], "files": files, "snapshot_revision": path.name,
              "model_type": cfg.model_type, "hidden_size": cfg.hidden_size, "api_calls": 0}
    run = Run(c)
    marker = run.file("diagnostics/model-download.json")
    write_json(marker, report)
    run.publish("model-download", [marker])
    return report


def stage_model(c, source_manifest):
    """Verify the offline transfer and install its pinned cache entry without copying weights."""
    import os
    from tempfile import TemporaryDirectory
    from transformers import AutoConfig
    from prism.v3.recovery import stage_lock

    require_runtime(c)
    run, stage = Run(c), "model-download"
    marker = run.file("diagnostics/model-download.json")
    with stage_lock(run, stage):
        if run.completed(stage):
            return read_json(marker)
        manifest = Path(source_manifest).resolve()
        value = read_json(manifest)
        if value["model"] != c["model"]:
            raise ValueError("Offline model transfer does not match the pinned model")
        files = value["files"]
        if any(Path(name).name != name or name.startswith(".") for name in files):
            raise ValueError("Offline model manifest contains unsafe paths")
        required = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
        if not required <= files.keys():
            raise ValueError("Offline model transfer lacks model/tokenizer assets")
        index = read_json(manifest.parent / "model.safetensors.index.json")
        if {name for name in files if name.endswith(".safetensors")} != set(index["weight_map"].values()):
            raise ValueError("Offline model transfer does not contain exactly the indexed weight shards")
        for name, spec in files.items():
            path = manifest.parent / name
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError(f"Offline model file differs from the verified local download: {name}")
        cfg = AutoConfig.from_pretrained(str(manifest.parent), local_files_only=True, trust_remote_code=False)
        if cfg.model_type != "olmo3" or cfg.hidden_size != 4096:
            raise ValueError("Wrong transferred architecture")
        repo = "models--" + c["model"]["id"].replace("/", "--")
        snapshots = Path(os.environ["PRISM_HF_CACHE_LOCATION"]) / "hub" / repo / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        revision = c["model"]["revision"]
        target = snapshots / revision
        if target.exists():
            for name, spec in files.items():
                if digest(target / name) != spec["sha256"]:
                    raise ValueError(f"Existing pinned snapshot file differs: {name}")
        else:
            with TemporaryDirectory(dir=snapshots, prefix=f".{revision}-") as directory:
                staging = Path(directory) / revision
                staging.mkdir()
                for name in files:
                    os.link(manifest.parent / name, staging / name)
                staging.rename(target)
        report = {**value, "snapshot_revision": revision, "model_type": cfg.model_type,
                  "hidden_size": cfg.hidden_size, "transfer_verified": True, "api_calls": 0}
        write_json(marker, report)
        run.publish(stage, [marker], [manifest])
        return report


def model_smoke(c, profile=0):
    import torch
    from prism.hf import generate, prompt_ids, sequence_stats as dense_stats
    from prism.v3.ppo import load_model, load_tokenizer
    from .models import sequence_stats
    require_runtime(c)
    run = Run(c)
    if profile < 0:
        raise ValueError("Model smoke profile must be nonnegative")
    stage = "model-smoke" if profile == 0 else f"model-smoke-{profile}"
    path = run.file(f"diagnostics/{stage}.json")
    if run.completed(stage):
        return read_json(path)
    model, tokenizer = load_model(c).requires_grad_(False), load_tokenizer(c)
    if model.config.model_type != "olmo3" or model.config.hidden_size != 4096:
        raise ValueError("Wrong V4 model architecture")
    if tokenizer.pad_token_id < model.config.vocab_size or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Sampling needs the external padding sentinel, separate from EOS")
    prompt = prompt_ids(tokenizer, [{"role": "user", "content": "Give a brief greeting."}], 2048)
    response = generate(model, tokenizer, prompt, 8, c["study"]["seed"])
    phi, ell = sequence_stats(model, prompt, response, tokenizer.eos_token_id)
    ref_phi, ref_ell = dense_stats(model, prompt, response, tokenizer.eos_token_id)
    if phi.shape != (4096,) or not np.isfinite(phi).all() or not np.isfinite(ell):
        raise ValueError("Nonfinite or wrong-dimensional EOS/logp")
    if not np.array_equal(phi, ref_phi) or abs(ell - ref_ell) > 1e-4:
        raise ValueError("Chunked scoring disagrees with the complete-vocabulary reference")
    report = {"passed": True, "model": c["model"], "dimension": 4096, "dtype": "float32",
              "chunked_dense_logp_error": abs(ell - ref_ell), "eos_feature_equal": True,
              "generated_tokens": len(response), "external_pad": tokenizer.pad_token_id,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "device_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
              "formal_ppo_episodes": 0, "api_calls": 0}
    write_json(path, report)
    inputs = [run.receipt_path("model-download")] if run.completed("model-download") else []
    run.publish(stage, [path], inputs)
    return report


def policy_smoke(c):
    """One real Adam update and a longest-sequence backward; formal policy stays closed."""
    import torch
    from prism.v3.ppo import load_model, load_tokenizer
    from prism.v3.ppo_trainer import PPOEngine, prepare_inputs, _trainable_state
    from .models import policy_models, response_logps, storage_report
    run = Run(c)
    run.require("model-smoke")
    if run.completed("policy-smoke"):
        return read_json(run.file("diagnostics/policy-smoke/report.json"))
    token = load_tokenizer(c)
    settings = deepcopy(c["ppo"])
    settings.update(total_episodes=2, rollout_batch_size=2, mini_batch_size=2,
                    gradient_accumulation_steps=2, num_mini_batches=1, num_ppo_epochs=1,
                    response_length=4, durable_checkpoint_interval=2, engineering_checkpoint=2,
                    checkpoints=[2], probability_check_rows=2)
    lm = load_model(c)
    w = np.linspace(-.01, .01, 4096, dtype=np.float32)
    actor, critic, reward = policy_models(lm, w, 0., 1., settings, eos=token.eos_token_id,
                                         pad=token.pad_token_id, context_tokens=c["data"]["context_tokens"])
    prompts = [token.apply_chat_template([{"role": "user", "content": x}], tokenize=True,
                                        add_generation_prompt=True) for x in ("Say hello.", "Say goodbye.")]
    before_reward = reward.score(prompts, [[token.eos_token_id]] * 2).clone()
    frozen = [(p, p._version) for p in reward.backbone.parameters()]
    before = [_trainable_state(m) for m in (actor, critic)]
    folder = run.file("diagnostics/policy-smoke/checkpoints")
    engine = PPOEngine(actor, critic, reward, prompts, pad=token.pad_token_id, eos=token.eos_token_id,
                       settings=settings, seed=11, contract_hash=run.fingerprint, checkpoint_dir=folder)
    result = engine.run()
    after = [_trainable_state(m) for m in (actor, critic)]
    if any(not any(not torch.equal(old[k], new[k]) for k in old) for old, new in zip(before, after)):
        raise ValueError("Actor or critic failed to update")
    # A worst-case configured 2048+512 sequence, not only the short rollout.
    context, length = c["data"]["max_prompt_tokens"], c["ppo"]["response_length"]
    ids = torch.full((1, context + length), prompts[0][-1], device=engine.device, dtype=torch.long)
    targets = ids[:, context:]
    engine.optimizer.zero_grad(set_to_none=True)
    loss = -response_logps(actor, prepare_inputs(ids, token.pad_token_id), targets, context,
                           chunk_size=settings["logp_chunk_tokens"]).mean()
    loss.backward()
    value = critic(**prepare_inputs(ids, token.pad_token_id))
    value.square().mean().backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in engine.groups.values() for p in group):
        raise ValueError("Nonfinite longest-sequence gradient")
    engine.optimizer.zero_grad(set_to_none=True)
    restored = PPOEngine(actor, critic, reward, prompts, pad=token.pad_token_id, eos=token.eos_token_id,
                         settings=settings, seed=11, contract_hash=run.fingerprint, checkpoint_dir=folder)
    restored.resume()
    if any(not all(torch.equal(saved[k], now[k]) for k in saved)
           for saved, now in zip(after, [_trainable_state(m) for m in (actor, critic)])):
        raise ValueError("Adapter/head checkpoint restore changed tensors")
    if any(p._version != version for p, version in frozen) or not torch.equal(before_reward, reward.score(prompts, [[token.eos_token_id]] * 2)):
        raise ValueError("Frozen Reward/base changed during PPO engineering update")
    report = {"passed": True, "model": c["model"], "storage": storage_report(actor, critic, reward.backbone),
              "actual_optimizer_steps": result["actual_optimizer_steps"], "engineering_episodes": 2,
              "longest_backward_tokens": context + length, "checkpoint_restored": True,
              "frozen_reward_unchanged": True, "formal_policy_admitted": False,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "api_calls": 0}
    path = run.file("diagnostics/policy-smoke/report.json")
    write_json(path, report)
    run.publish("policy-smoke", [path, *sorted(folder.glob("*.pt")), folder / "latest.json"],
                inputs=[run.receipt_path("model-smoke")])
    return report


def cohort_audit(c, reference_run):
    """Compare canonical identities/labels, never model-specific token counts."""
    if not reference_run:
        raise ValueError("cohort-audit needs --reference-run /absolute/path/to/v3-run")
    reference = Path(reference_run).resolve()
    run = Run(c)
    run.require("data")
    marker = read_json(reference / "experiment.json")
    if marker["protocol"] != "hh_finite_pool_v3":
        raise ValueError("Reference must be the original V3 HH experiment")
    receipt = read_json(reference / "receipts/data.json")
    result, dependencies = {}, [run.receipt_path("data"), reference / "experiment.json", reference / "receipts/data.json"]
    for name in ("pool", "labels", "policy", "ppo_dev", "evaluation", "reward_dev"):
        other, current = reference / f"data/{name}.jsonl", run.file(f"data/{name}.jsonl")
        expected = receipt["outputs"].get(f"run:data/{name}.jsonl")
        if expected != digest(other):
            raise ValueError("Reference cohort is not bound to its original receipt")
        old, new = rows(other), rows(current)
        def identity(r):
            return {k: v for k, v in r.items() if k not in ("prompt_tokens", "answer_tokens", "total_tokens")}
        if list(map(identity, old)) != list(map(identity, new)):
            raise ValueError(f"V4 {name} identities/labels differ; resolve cohort comparability before training")
        result[name] = len(old)
        dependencies.append(other)
    report = {"passed": True, "counts": result, "reference_config_sha256": marker["config_sha256"],
              "reused_token_ids": False, "reused_features_or_geometry": False}
    path = run.file("diagnostics/cohort-audit.json")
    write_json(path, report)
    run.publish("cohort-audit", [path], inputs=dependencies)
    return report
