"""Real tiny OLMo2 + real PEFT LoRA integration; never downloads model weights."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Olmo2ForCausalLM"):
    pytest.skip("Tiny OLMo2 requires the pinned Transformers runtime", allow_module_level=True)

from transformers import Olmo2Config, Olmo2ForCausalLM
from prism.v3.ppo import FrozenNormalizedReward, make_actor, make_critic, prefit_critic
from prism.v3.ppo_math import gae_reference
from prism.v3.ppo_trainer import (PPOEngine, PromptSampler, gae, left_pad, optimizer_steps,
                                  prepare_inputs, response_masks, seed_everything, stream_generate)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(original)


def settings(micro=1):
    return {"total_episodes": 12, "rollout_batch_size": 4, "mini_batch_size": 2,
            "per_device_train_batch_size": micro, "gradient_accumulation_steps": 2 // micro,
            "num_mini_batches": 2, "num_ppo_epochs": 2, "learning_rate": .002,
            "critic_learning_rate": .003, "critic_head_learning_rate": .004,
            "lora_r": 2, "lora_alpha": 4, "lora_target_modules": "all-linear", "lora_dropout": 0.,
            "critic_lora_r": 2, "critic_lora_alpha": 4,
            "warmup_ratio": .05, "minimum_learning_rate_ratio": .1,
            "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-5, "weight_decay": 0.,
            "gamma": 1., "lam": .95, "kl_coef": .05, "cliprange": .2, "cliprange_value": .2,
            "vf_coef": .1, "max_grad_norm": .03, "response_length": 4, "temperature": 1.,
            "whiten_rewards": False, "whiten_advantages": True,
            "local_rollout_forward_batch_size": 2, "probability_check_rows": 4,
            "max_logp_error": 1e-3, "max_zero_update_approxkl": 1e-7,
            "engineering_checkpoint": 4, "checkpoints": [8, 12],
            "durable_checkpoint_interval": 4,
            "critic_warmup_episodes": 4, "critic_warmup_epochs": 4, "critic_warmup_mini_batch_size": 2}


def models():
    seed_everything(12345)
    configuration = Olmo2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                               max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
                               pad_token_id=0, attention_dropout=0., tie_word_embeddings=False)
    configuration._attn_implementation = "eager"
    lm = Olmo2ForCausalLM(configuration).float().eval()
    value_base, reward_base = deepcopy(lm.model), deepcopy(lm.model)
    p = settings()
    weights = np.linspace(-.5, .5, 16, dtype=np.float32)
    actor = make_actor(lm, p)
    critic = make_critic(value_base, weights, .3, 1.2, p)
    reward = FrozenNormalizedReward(reward_base, weights, .3, 1.2, eos=2, pad=37, context_tokens=64)
    return actor, critic, reward


def engine(path=None, micro=1, contract="fixture"):
    actor, critic, reward = models()
    prompts = [[1, 5, 6], [1, 7], [1, 8, 9, 10], [1, 11]]
    return PPOEngine(actor, critic, reward, prompts, pad=37, eos=2, settings=settings(micro),
                     seed=11, contract_hash=contract, checkpoint_dir=path)


def trainable(model):
    return {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}


def test_real_olmo2_streaming_cached_and_full_paths_agree():
    actor, _, _ = models()
    prompts = left_pad([[1, 5, 6], [1, 7]], 37, "cpu")
    seed_everything(11)
    responses, selected = stream_generate(actor, prompts, pad=37, eos=2, max_new_tokens=6)
    packed = torch.cat((prompts, responses), 1)
    with torch.no_grad():
        output = actor(**prepare_inputs(packed, 37))
        logits = output.logits[:, prompts.shape[1] - 1:-1]
        direct = logits.log_softmax(-1).gather(-1, responses.masked_fill(responses.eq(37), 0).unsqueeze(-1)).squeeze(-1)
    assert torch.max(torch.abs((selected - direct)[responses.ne(37)])) < 1e-5
    assert selected.shape == responses.shape
    assert selected.ndim == 2  # no retained batch*time*vocabulary tensor


def test_gae_and_terminal_post_eos_masks_match_numpy_reference():
    response = torch.tensor([[9, 2, 37, 37], [7, 8, 9, 10]])
    valid, value_valid, length = response_masks(response, 37)
    assert valid.tolist() == [[True, True, False, False], [True] * 4]
    assert value_valid.tolist() == [[True, True, True, False], [True] * 4]
    assert length.tolist() == [2, 4]
    reward = torch.tensor([[0., 0., 2., 0.], [0., .2, 0., 1.]])
    value = torch.tensor([[.1, .2, .3, 0.], [.2, .4, .1, .2]])
    actual = gae(reward, value, 1., .95)
    reference = gae_reference(reward.numpy(), value.numpy(), 1., .95)
    assert np.allclose(actual[0].numpy(), reference[0], atol=1e-6)
    assert np.allclose(actual[1].numpy(), reference[1], atol=1e-6)


def test_frozen_reward_parity_natural_and_feature_only_eos():
    _, critic, reward = models()
    assert not ({id(p) for p in critic.parameters()} & {id(p) for p in reward.backbone.parameters()})
    prompts, responses = [[1, 3], [1, 4]], [[7, 2], [8, 9]]
    score = reward.score(prompts, responses)
    direct = []
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            sequence = prompt + response
            if sequence[-1] != 2:
                sequence.append(2)
            tokens = torch.tensor([sequence])
            phi = reward.backbone(**prepare_inputs(tokens, 37)).last_hidden_state[0, -1]
            direct.append((phi @ reward.weights - reward.mean) / reward.std)
    assert torch.allclose(score, torch.stack(direct), atol=1e-6)


def test_actual_adamw_updates_clipping_and_frozen_reference():
    subject = engine()
    before_actor, before_critic = trainable(subject.actor), trainable(subject.critic)
    result = subject.run()
    assert result["completed"] and result["completed_episodes"] == 12
    assert all(v["step_min"] == v["step_max"] == 12 for v in result["actual_optimizer_steps"].values())
    assert any(not torch.equal(before_actor[k], p) for k, p in trainable(subject.actor).items())
    assert any(not torch.equal(before_critic[k], p) for k, p in trainable(subject.critic).items())
    assert all(row["zero_update_approxkl"] < 1e-7 for row in result["probability_checks"])
    assert any(row["critic_gradient_norm"] > subject.settings["max_grad_norm"] for row in result["history"])
    for role in ("actor", "critic_lora", "critic_head"):
        assert all(torch.isfinite(p).all() for p in subject.groups[role])
    actor_norm = torch.stack([p.grad.square().sum() for p in subject.groups["actor"] if p.grad is not None]).sum().sqrt()
    critic_norm = torch.stack([p.grad.square().sum() for role in ("critic_lora", "critic_head") for p in subject.groups[role] if p.grad is not None]).sum().sqrt()
    assert actor_norm <= subject.settings["max_grad_norm"] + 1e-6
    assert critic_norm <= subject.settings["max_grad_norm"] + 1e-6
    assert all(parameter._version == version for parameter, version in subject._frozen)


def test_checkpoint_resume_is_bitwise_equal_to_uninterrupted_training(tmp_path):
    continuous = engine()
    continuous.run()
    interrupted = engine(tmp_path / "checkpoint")
    stopped = interrupted.run(stop_after_episodes=4)
    assert not stopped["completed"]
    resumed = engine(tmp_path / "checkpoint")
    resumed.resume()
    result = resumed.run()
    assert result["completed"]
    assert resumed.history == continuous.history
    for name, tensor in trainable(continuous.actor).items():
        assert torch.equal(tensor, trainable(resumed.actor)[name])
    for name, tensor in trainable(continuous.critic).items():
        assert torch.equal(tensor, trainable(resumed.critic)[name])
    with pytest.raises(ValueError, match="different"):
        engine(tmp_path / "checkpoint", contract="different").resume()


def test_microbatch_accumulation_preserves_whole_minibatch_objective():
    first = engine(micro=1)
    seed_everything(111)
    batch = first.rollout()
    second = engine(micro=2)
    first.update(batch)
    second.update(batch)
    for name, tensor in trainable(first.actor).items():
        assert torch.allclose(tensor, trainable(second.actor)[name], atol=2e-6, rtol=1e-5)
    for name, tensor in trainable(first.critic).items():
        assert torch.allclose(tensor, trainable(second.critic)[name], atol=2e-6, rtol=1e-5)


def test_critic_prefit_uses_prefix_values_and_counts_real_steps():
    _, critic, reward = models()
    records = [{"id": str(i), "prompt_tokens": [1, 3 + i], "response_tokens": [8 + i, 2]} for i in range(6)]
    result = prefit_critic(critic, reward, records[:4], records[4:], settings=settings(), pad=37, seed=11)
    assert result["actual_optimizer_steps"] == 8
    assert result["episodes"] == 4 and result["heldout_trajectories"] == 2
    assert np.isfinite(result["after"]["mse"])
    assert result["gradient_norm_clipping_executed"]


def test_sampler_replay_preserves_epoch_and_cursor():
    sampler = PromptSampler(8, 11)
    assert len(set(sampler.take(4))) == 4
    clone = PromptSampler(8, 999)
    clone.load_state_dict(sampler.state_dict())
    assert clone.take(4) == sampler.take(4)
    assert clone.take(4) == sampler.take(4)


def test_probe_callback_does_not_perturb_training_rng():
    import random
    continuous = engine()
    continuous.run()
    with_callback = engine()

    def consume_rng(subject):
        random.random()
        np.random.random(11)
        torch.rand(13)

    with_callback.run(on_checkpoint=consume_rng)
    assert with_callback.history == continuous.history
    for name, value in trainable(continuous.actor).items():
        assert torch.equal(value, trainable(with_callback.actor)[name])


def test_shared_sft_progress_recovers_without_regenerating_finished_items(tmp_path, monkeypatch):
    import prism.v3.ppo as stage
    from prism.v3.artifacts import Run
    from prism.io import rows

    class Tokenizer:
        pad_token_id, eos_token_id = 37, 2

        def apply_chat_template(self, messages, **kwargs):
            return [1, 5 + int(messages[-1]["content"])]

    p = settings()
    p["fixed_train_probe_prompts"] = 2
    c = {"output": str(tmp_path / "resumed"), "study": {"seed": 11},
         "data": {"max_prompt_tokens": 16, "context_tokens": 64, "feature_dimension": 16},
         "reward": {"calibration_responses": 6}, "ppo": p}
    model = models()[0].get_base_model()
    monkeypatch.setattr(stage, "load_model", lambda config: deepcopy(model))
    monkeypatch.setattr(stage, "load_tokenizer", lambda config: Tokenizer())

    def prepare_data(config):
        run = Run(config)
        policy = [{"id": str(i), "messages": [{"role": "user", "content": str(i)}]} for i in range(8)]
        development = [{"id": str(i), "messages": [{"role": "user", "content": str(i)}]} for i in (10, 11)]
        stage._save_rows(run.file("data/policy.jsonl"), policy)
        stage._save_rows(run.file("data/ppo_dev.jsonl"), development)
        run.publish("data", [run.file("data/policy.jsonl"), run.file("data/ppo_dev.jsonl")])
        return run

    subject = prepare_data(c)
    original = stream_generate
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("simulated worker exit")
        return original(*args, **kwargs)

    monkeypatch.setattr("prism.v3.ppo_trainer.stream_generate", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        stage.prepare_shared(c)
    prefix = rows(subject.file("shared/generation.progress.jsonl"))
    assert len(prefix) == 2
    monkeypatch.setattr("prism.v3.ppo_trainer.stream_generate", original)
    stage.prepare_shared(c)
    assert rows(subject.file("shared/generation.progress.jsonl"))[:2] == prefix
    assert len(rows(subject.file("shared/calibration.jsonl"))) == 6
    assert len(rows(subject.file("shared/train_probe.jsonl"))) == 2
    assert len(rows(subject.file("shared/dev_probe.jsonl"))) == 2
    full_c = deepcopy(c)
    full_c["output"] = str(tmp_path / "continuous")
    full = prepare_data(full_c)
    stage.prepare_shared(full_c)
    assert rows(subject.file("shared/calibration.jsonl")) == rows(full.file("shared/calibration.jsonl"))
    with np.load(subject.file("shared/calibration.features.npz")) as resumed, np.load(full.file("shared/calibration.features.npz")) as continuous:
        assert np.array_equal(resumed["phi"], continuous["phi"])
    monkeypatch.setattr(stage, "load_model", lambda config: pytest.fail("Completed shared stage loaded a model"))
    stage.prepare_shared(c)


@pytest.mark.parametrize("failure", ["before_dev_json", "before_dev_receipt"])
def test_checkpoint_probe_callback_crash_is_completed_on_resume(tmp_path, monkeypatch, failure):
    """Commit checkpoint, lose worker during callback, finish exact probes first."""
    import prism.v3.ppo as stage
    from prism.v3.artifacts import Run, write_json
    from prism.io import digest
    c = {"output": str(tmp_path / "run"), "ppo": settings()}
    run = Run(c)
    subject = engine(run.file("policy/empirical/checkpoints"))
    source, outputs = [], []
    for name in ("train_probe", "dev_probe"):
        records = [{"id": name + str(i), "source_id": str(i), "seed": i + 100,
                    "prompt_tokens": [1, 5 + i], "response_tokens": [7 + i, 2]} for i in range(2)]
        phi = []
        with torch.no_grad():
            for row in records:
                tokens = torch.tensor([row["prompt_tokens"] + row["response_tokens"]])
                phi.append(subject.reward.backbone(**prepare_inputs(tokens, 37)).last_hidden_state[0, -1].numpy())
        record_path, feature_path = run.file(f"shared/{name}.jsonl"), run.file(f"shared/{name}.features.npz")
        stage._save_rows(record_path, records)
        stage._save_features(feature_path, phi, [row["id"] for row in records])
        outputs.extend((record_path, feature_path))
    run.publish("shared-sft", outputs)
    norm_path = run.file("normalization/empirical.json")
    write_json(norm_path, {"mean": subject.reward.mean, "std": subject.reward.std})
    run.publish("normalization", [norm_path])
    weights = run.file("reward/empirical.npz")
    weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(weights, w=subject.reward.weights.numpy())
    write_json(run.file("policy/empirical/contract.json"), {"contract_hash": "fixture"})
    original_publish = Run.publish

    def publish_with_failure(self, name, *args, **kwargs):
        if failure == "before_dev_receipt" and name.endswith("dev_probe"):
            raise RuntimeError("worker lost after probe JSON")
        return original_publish(self, name, *args, **kwargs)

    def interrupted(instance):
        stage.evaluate_probe(c, instance, "empirical", run, "train_probe")
        if failure == "before_dev_json":
            raise RuntimeError("worker lost before probe JSON")
        stage.evaluate_probe(c, instance, "empirical", run, "dev_probe")

    monkeypatch.setattr(Run, "publish", publish_with_failure)
    with pytest.raises(RuntimeError, match="worker lost"):
        subject.run(stop_after_episodes=4, on_checkpoint=interrupted)
    assert run.file("policy/empirical/checkpoints/episode-00000004.pt").is_file()
    finished_probe = run.file("policy/empirical/probes/00000004-train_probe.json")
    old_digest = digest(finished_probe)
    incomplete = run.file("policy/empirical/probes/00000004-dev_probe.json")
    incomplete_digest = digest(incomplete) if incomplete.exists() else None
    monkeypatch.setattr(Run, "publish", original_publish)

    def complete(instance):
        for name in ("train_probe", "dev_probe"):
            stage.evaluate_probe(c, instance, "empirical", run, name)

    resumed = engine(run.file("policy/empirical/checkpoints"))
    resumed.resume(on_checkpoint=complete)
    assert run.completed("probe-empirical-00000004-dev_probe")
    assert digest(finished_probe) == old_digest
    if incomplete_digest is not None:
        assert digest(incomplete) == incomplete_digest
    resumed.run(on_checkpoint=complete)
    continuous = engine()
    continuous.run()
    assert resumed.history == continuous.history
    for name, value in trainable(continuous.actor).items():
        assert torch.equal(value, trainable(resumed.actor)[name])
    with incomplete.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    # A previously published probe must be hash-verified, not silently rewritten.
    resumed.completed_episodes = 4
    with pytest.raises(ValueError, match="changed"):
        stage.evaluate_probe(c, resumed, "empirical", run, "dev_probe")


def test_durable_checkpoints_do_not_trigger_extra_probes_and_resume_without_replay(tmp_path):
    from prism.io import digest
    from prism.v3.artifacts import write_json
    continuous = engine(tmp_path / "continuous")
    continuous.settings["engineering_checkpoint"] = 8
    continuous.settings["checkpoints"] = [12]
    calls = []
    result = continuous.run(on_checkpoint=lambda subject: calls.append(subject.completed_episodes))
    assert calls == [8, 12]
    assert result["durable_checkpoint_interval"] == 4
    early = tmp_path / "continuous/episode-00000004.pt"
    assert early.is_file()
    state = torch.load(early, weights_only=True)
    assert state["diagnostics_due"] is False and state["episode"] == 4
    checkpoint_folder = tmp_path / "resumed"
    checkpoint_folder.mkdir()
    import shutil
    shutil.copyfile(early, checkpoint_folder / early.name)
    write_json(checkpoint_folder / "latest.json", {"path": early.name, "sha256": digest(early),
               "contract_hash": "fixture", "episode": 4, "diagnostics_due": False})
    resumed = engine(checkpoint_folder)
    resumed.settings["engineering_checkpoint"] = 8
    resumed.settings["checkpoints"] = [12]
    resumed_calls = []
    resumed.resume(on_checkpoint=lambda subject: resumed_calls.append(subject.completed_episodes))
    assert resumed_calls == []
    resumed.run(on_checkpoint=lambda subject: resumed_calls.append(subject.completed_episodes))
    assert resumed_calls == [8, 12]
    assert resumed.history == continuous.history
    for name, value in trainable(continuous.actor).items():
        assert torch.equal(value, trainable(resumed.actor)[name])


def test_shared_shards_match_serial_and_resume_exactly(tmp_path, monkeypatch):
    import prism.v3.ppo as stage
    import prism.v3.shared as sharded
    from prism.v3.artifacts import Run
    from prism.io import rows

    class Tokenizer:
        pad_token_id, eos_token_id = 37, 2

        def apply_chat_template(self, messages, **kwargs):
            return [1, 5 + int(messages[-1]["content"])]

    p = settings()
    p["fixed_train_probe_prompts"] = 2
    c = {"output": str(tmp_path / "shards"), "study": {"seed": 11},
         "data": {"max_prompt_tokens": 16, "context_tokens": 64, "feature_dimension": 16},
         "reward": {"calibration_responses": 6}, "ppo": p}
    model = models()[0].get_base_model()
    monkeypatch.setattr(stage, "load_model", lambda config: deepcopy(model))
    monkeypatch.setattr(stage, "load_tokenizer", lambda config: Tokenizer())

    def prepare(config):
        run = Run(config)
        for name, indices in (("policy", range(8)), ("ppo_dev", (10, 11))):
            stage._save_rows(run.file(f"data/{name}.jsonl"),
                [{"id": str(i), "messages": [{"role": "user", "content": str(i)}]} for i in indices])
        run.publish("data", [run.file("data/policy.jsonl"), run.file("data/ppo_dev.jsonl")])
        return run

    run = prepare(c)
    original = sharded._generate_row
    calls = []

    def interrupted(*args):
        calls.append(args[-1])
        if len(calls) == 2:
            raise RuntimeError("simulated shard worker exit")
        return original(*args)

    monkeypatch.setattr(sharded, "_generate_row", interrupted)
    with pytest.raises(RuntimeError, match="simulated shard"):
        sharded.prepare_shared_part(c, 0, 3)
    assert calls == [0, 3]
    saved = rows(run.file("shared/parts/00000-of-00003/progress.jsonl"))
    resumed = []

    def resume(*args):
        resumed.append(args[-1])
        return original(*args)

    monkeypatch.setattr(sharded, "_generate_row", resume)
    for shard in (0, 2, 1):
        sharded.prepare_shared_part(c, shard, 3)
    assert 0 not in resumed
    assert rows(run.file("shared/parts/00000-of-00003/progress.jsonl"))[:1] == saved
    sharded.merge_shared(c, 3)
    serial_c = deepcopy(c)
    serial_c["output"] = str(tmp_path / "serial")
    serial = prepare(serial_c)
    stage.prepare_shared(serial_c)
    for group in ("calibration", "dev_probe", "train_probe"):
        assert run.file(f"shared/{group}.jsonl").read_bytes() == serial.file(f"shared/{group}.jsonl").read_bytes()
        with np.load(run.file(f"shared/{group}.features.npz")) as a, np.load(serial.file(f"shared/{group}.features.npz")) as b:
            assert np.array_equal(a["phi"], b["phi"])
            assert np.array_equal(a["ids"], b["ids"])
    smoke = sharded.shared_smoke(c)
    assert smoke["metadata"]["actual_generated_responses"] == 12
    assert smoke["metadata"]["max_feature_error"] == 0
    monkeypatch.setattr(stage, "load_model", lambda c: pytest.fail("completed shard loaded model"))
    sharded.prepare_shared_part(c, 0, 3)
    sharded.merge_shared(c, 3)
    # Hash verification remains active even after successful publication.
    with run.file("shared/parts/00000-of-00003/progress.jsonl").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="changed"):
        sharded.prepare_shared_part(c, 0, 3)
