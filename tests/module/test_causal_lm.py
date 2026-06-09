import pytest
import torch

from model.config import CTMLLMConfig


def _causal_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=128,
        num_hidden_layers=1,
        d_model=64,
        d_input=32,
        iterations=2,
        memory_length=4,
        memory_hidden_dims=4,
        deep_nlms=False,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        self_cond=False,
        cross_layer_state=False,
        dropout=0.0,
        tick_halt_mode="none",
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


class TestCTMForCausalLMForward:
    def test_inference_forward(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(ids)
        assert out["logits"].shape == (2, 8, cfg.vocab_size)
        assert out["loss"] is None

    def test_inference_with_labels(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(ids, labels=ids)
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"])

    def test_forward_train_finite_loss(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, losses, certainties = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)
        assert losses.shape == (2, 2)
        assert certainties.shape == (2, 2)

    def test_backward(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(g > 0 for g in grad_norms)


class TestCTMForCausalLMTickLossModes:
    @pytest.mark.parametrize("mode", ["min_conf", "mean", "last"])
    def test_tick_loss_modes(self, mode):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(tick_loss_mode=mode, iterations=3)
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, losses, certainties = model.forward_train(ids, ids, num_iters=3)
        assert torch.isfinite(loss)
        assert losses.shape[1] == 3


class TestCTMForCausalLMELFHorizon:
    @pytest.mark.parametrize("mode", ["none", "linear", "pow2"])
    def test_elf_horizon_modes(self, mode):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(elf_horizon_mode=mode, elf_max_horizon=2, iterations=2)
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)


class TestCTMForCausalLMHalt:
    def test_halt_confidence_mode(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(
            tick_halt_mode="confidence",
            tick_halt_threshold=0.3,
            tick_halt_temperature=0.25,
            iterations=4,
        )
        model = CTMForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, losses, certainties = model.forward_train(ids, ids, num_iters=4)
        assert torch.isfinite(loss)

    def test_halt_threshold_mode(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(
            tick_halt_mode="threshold",
            tick_halt_threshold=0.5,
            iterations=4,
        )
        model = CTMForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, losses, _ = model.forward_train(ids, ids, num_iters=4)
        assert torch.isfinite(loss)


class TestCTMForCausalLMGeneration:
    def test_generate_basic(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=4, num_iters=2)
        assert generated.shape == (1, 8)

    def test_generate_with_topk_topp(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(
            ids, max_new_tokens=4, top_k=10, top_p=0.9, num_iters=2)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 4

    def test_generate_repetition_penalty(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config()
        model = CTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(
            ids, max_new_tokens=4, repetition_penalty=1.2, num_iters=2)
        assert generated.shape == (1, 8)


class TestCTMForCausalLMFastSlow:
    def test_reflex_mode(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(
            fast_output_mode="reflex",
            fast_output_weight=0.1,
        )
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)

    def test_anytime_mode(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(
            fast_output_mode="anytime",
            fast_output_weight=0.1,
            habit_output_weight=0.1,
            fast_output_ticks="1,2",
        )
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)


class TestCTMForCausalLMDINO:
    def test_dino_enabled(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(dino_self_supervised_weight=0.1)
        model = CTMForCausalLM(cfg)
        assert model.dino_enabled
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)
        assert model.last_dino_loss != 0.0


class TestCTMForCausalLMTTT:
    def test_ttt_layer(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(ttt_layer=True)
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)


class TestCTMForCausalLMContextReading:
    def test_fusion_mode(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(
            context_reading_mode="fusion",
            context_reading_sources="local,egram",
        )
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)

    def test_all_sources(self):
        from model.model_ctm_llm import CTMForCausalLM
        cfg = _causal_config(
            context_reading_mode="fusion",
            context_reading_sources="local,compressed,retrieval,expert,egram",
        )
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)


class TestBuildCTMForCausalLM:
    def test_sync_default(self):
        from model.model_ctm_llm import build_ctm_for_causal_lm, CTMForCausalLM
        cfg = _causal_config()
        model = build_ctm_for_causal_lm(cfg)
        assert isinstance(model, CTMForCausalLM)

    def test_async_banded(self):
        from model.model_ctm_llm import build_ctm_for_causal_lm
        from model.model_ctm_async import AsyncCTMForCausalLM
        cfg = _causal_config(async_tick_mode="banded", async_tick_periods="1,2")
        model = build_ctm_for_causal_lm(cfg)
        assert isinstance(model, AsyncCTMForCausalLM)
