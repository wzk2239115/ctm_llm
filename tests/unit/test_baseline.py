import pytest
import torch
import numpy as np

from baseline.models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES,
)
from baseline.models.ctm import ContinuousThoughtMachine


def _base_params(**overrides):
    p = dict(
        iterations=4,
        d_model=32,
        d_input=4,
        heads=2,
        n_synch_out=3,
        n_synch_action=3,
        synapse_depth=1,
        memory_length=5,
        deep_nlms=True,
        memory_hidden_dims=2,
        do_layernorm_nlm=False,
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type="first-last",
        n_random_pairing_self=0,
    )
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# Core CTM construction
# ---------------------------------------------------------------------------


class TestContinuousThoughtMachineConstruction:
    def test_default_construction(self):
        model = ContinuousThoughtMachine(**_base_params())
        assert model.iterations == 4
        assert model.d_model == 32
        assert model.out_dims == 10

    @pytest.mark.parametrize("neuron_select_type", VALID_NEURON_SELECT_TYPES)
    def test_all_neuron_select_types(self, neuron_select_type):
        params = _base_params(neuron_select_type=neuron_select_type)
        model = ContinuousThoughtMachine(**params)
        assert model.neuron_select_type == neuron_select_type

    def test_invalid_neuron_select_type_raises(self):
        with pytest.raises(Exception):
            ContinuousThoughtMachine(**_base_params(neuron_select_type="invalid"))

    def test_backbone_none_no_positional_embedding(self):
        model = ContinuousThoughtMachine(**_base_params(
            backbone_type="none",
            positional_embedding_type="none",
        ))
        assert model is not None

    @pytest.mark.parametrize("pos_emb", VALID_POSITIONAL_EMBEDDING_TYPES)
    def test_backbone_none_with_positional_embedding_raises(self, pos_emb):
        with pytest.raises(Exception):
            ContinuousThoughtMachine(**_base_params(
                backbone_type="none",
                positional_embedding_type=pos_emb,
            ))

    @pytest.mark.parametrize("backbone_type", ["parity_backbone", "shallow-wide"])
    @pytest.mark.parametrize("pos_emb", ["none"] + VALID_POSITIONAL_EMBEDDING_TYPES)
    def test_valid_backbone_with_positional_embeddings(self, backbone_type, pos_emb):
        d_input = 16
        model = ContinuousThoughtMachine(**_base_params(
            backbone_type=backbone_type,
            positional_embedding_type=pos_emb,
            d_input=d_input,
        ))
        assert model is not None

    def test_deep_vs_shallow_nlm(self):
        deep = ContinuousThoughtMachine(**_base_params(deep_nlms=True))
        shallow = ContinuousThoughtMachine(**_base_params(deep_nlms=False))
        deep_params = sum(p.numel() for p in deep.trace_processor.parameters())
        shallow_params = sum(p.numel() for p in shallow.trace_processor.parameters())
        assert deep_params > shallow_params

    def test_synapse_depth_1_vs_deep(self):
        shallow = ContinuousThoughtMachine(**_base_params(synapse_depth=1))
        from baseline.models.modules import SynapseUNET
        deep = ContinuousThoughtMachine(**_base_params(synapse_depth=2))
        assert not isinstance(shallow.synapses, SynapseUNET)
        assert isinstance(deep.synapses, SynapseUNET)

    def test_synch_representation_size_first_last(self):
        n = 4
        model = ContinuousThoughtMachine(**_base_params(
            neuron_select_type="first-last",
            n_synch_out=n,
            n_synch_action=n,
        ))
        expected = n * (n + 1) // 2
        assert model.synch_representation_size_out == expected

    def test_synch_representation_size_random_pairing(self):
        n = 5
        model = ContinuousThoughtMachine(**_base_params(
            neuron_select_type="random-pairing",
            n_synch_out=n,
            n_synch_action=n,
            n_random_pairing_self=2,
        ))
        assert model.synch_representation_size_out == n

    def test_decay_params_registered(self):
        model = ContinuousThoughtMachine(**_base_params())
        assert hasattr(model, "decay_params_out")
        assert hasattr(model, "decay_params_action")
        assert model.decay_params_out.requires_grad

    def test_neuron_indices_are_buffers(self):
        model = ContinuousThoughtMachine(**_base_params())
        assert isinstance(model.out_neuron_indices_left, torch.Tensor)
        assert isinstance(model.out_neuron_indices_right, torch.Tensor)
        assert model.out_neuron_indices_left.dtype == torch.long


# ---------------------------------------------------------------------------
# Core CTM forward pass
# ---------------------------------------------------------------------------


class TestContinuousThoughtMachineForward:
    def _make_parity_model(self, **kw):
        parity_length = 8
        params = _base_params(
            backbone_type="parity_backbone",
            positional_embedding_type="custom-rotational-1d",
            out_dims=2 * parity_length,
            prediction_reshaper=[parity_length, 2],
            d_input=8,
            **kw,
        )
        return ContinuousThoughtMachine(**params), parity_length

    def test_parity_forward_shapes(self):
        model, pl = self._make_parity_model()
        x = torch.randint(0, 2, (2, pl)).float() * 2 - 1
        preds, certs, synch_out = model(x)
        B = 2
        assert preds.shape == (B, 2 * pl, model.iterations)
        assert certs.shape == (B, 2, model.iterations)

    def test_parity_no_nans(self):
        model, pl = self._make_parity_model()
        x = torch.randint(0, 2, (2, pl)).float() * 2 - 1
        preds, certs, _ = model(x)
        assert not torch.isnan(preds).any()
        assert not torch.isnan(certs).any()

    def test_forward_with_tracking(self):
        model, pl = self._make_parity_model()
        model.eval()
        x = torch.randint(0, 2, (1, pl)).float() * 2 - 1
        result = model(x, track=True)
        assert len(result) == 6
        preds, certs, (synch_out, synch_act), pre_act, post_act, attn = result
        assert pre_act.shape[0] == model.iterations
        assert post_act.shape[0] == model.iterations

    @pytest.mark.parametrize("neuron_select_type", VALID_NEURON_SELECT_TYPES)
    def test_forward_all_neuron_select_types(self, neuron_select_type):
        model, pl = self._make_parity_model(
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=1,
        )
        x = torch.randint(0, 2, (2, pl)).float() * 2 - 1
        preds, certs, _ = model(x)
        assert not torch.isnan(preds).any()

    def test_gradient_flows(self):
        model, pl = self._make_parity_model()
        x = torch.randint(0, 2, (2, pl)).float() * 2 - 1
        preds, _, _ = model(x)
        loss = preds.sum()
        loss.backward()
        grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_shallow_wide_backbone_forward(self):
        params = _base_params(
            backbone_type="shallow-wide",
            positional_embedding_type="none",
            d_input=16,
        )
        model = ContinuousThoughtMachine(**params)
        x = torch.randn(2, 1, 28, 28)
        preds, certs, _ = model(x)
        assert preds.shape == (2, 10, model.iterations)

    @pytest.mark.parametrize("backbone,pos_emb", [
        ("shallow-wide", "learnable-fourier"),
        ("shallow-wide", "custom-rotational"),
        ("parity_backbone", "custom-rotational-1d"),
    ])
    def test_backbone_positional_combinations_forward(self, backbone, pos_emb):
        d_input = 8
        params = _base_params(
            backbone_type=backbone,
            positional_embedding_type=pos_emb,
            d_input=d_input,
        )
        model = ContinuousThoughtMachine(**params)
        if backbone == "parity_backbone":
            x = torch.randint(0, 2, (2, 8)).float() * 2 - 1
        else:
            x = torch.randn(2, 1, 28, 28)
        preds, _, _ = model(x)
        assert not torch.isnan(preds).any()


# ---------------------------------------------------------------------------
# QAMNIST variant
# ---------------------------------------------------------------------------


class TestCTMQAMNIST:
    def test_construction(self):
        from baseline.models.ctm_qamnist import ContinuousThoughtMachineQAMNIST
        params = dict(
            iterations=2,
            d_model=32,
            d_input=8,
            heads=2,
            n_synch_out=4,
            n_synch_action=4,
            synapse_depth=1,
            memory_length=4,
            deep_nlms=True,
            memory_hidden_dims=4,
            do_layernorm_nlm=False,
            out_dims=10,
            iterations_per_digit=1,
            iterations_per_question_part=1,
            iterations_for_answering=1,
            neuron_select_type="first-last",
        )
        model = ContinuousThoughtMachineQAMNIST(**params)
        assert model.iterations_per_digit == 1

    def test_forward_shape(self):
        from baseline.models.ctm_qamnist import ContinuousThoughtMachineQAMNIST
        params = dict(
            iterations=1,
            d_model=32,
            d_input=8,
            heads=2,
            n_synch_out=4,
            n_synch_action=4,
            synapse_depth=1,
            memory_length=4,
            deep_nlms=True,
            memory_hidden_dims=4,
            do_layernorm_nlm=False,
            out_dims=10,
            iterations_per_digit=2,
            iterations_per_question_part=1,
            iterations_for_answering=1,
            neuron_select_type="first-last",
        )
        model = ContinuousThoughtMachineQAMNIST(**params)
        B = 2
        x = torch.randn(B, 2, 1, 28, 28)
        z = torch.tensor([[-1, -2], [-2, -1]])
        preds, certs, synch = model(x, z)
        T = 2 + 2 + 1
        assert preds.shape == (B, 10, T)
        assert certs.shape == (B, 2, T)

    def test_no_nans(self):
        from baseline.models.ctm_qamnist import ContinuousThoughtMachineQAMNIST
        params = dict(
            iterations=1,
            d_model=32,
            d_input=8,
            heads=2,
            n_synch_out=4,
            n_synch_action=4,
            synapse_depth=1,
            memory_length=4,
            deep_nlms=True,
            memory_hidden_dims=4,
            do_layernorm_nlm=False,
            out_dims=10,
            iterations_per_digit=1,
            iterations_per_question_part=1,
            iterations_for_answering=1,
            neuron_select_type="first-last",
        )
        model = ContinuousThoughtMachineQAMNIST(**params)
        x = torch.randn(1, 1, 1, 28, 28)
        z = torch.tensor([[-1, -2]])
        preds, _, _ = model(x, z)
        assert not torch.isnan(preds).any()


# ---------------------------------------------------------------------------
# SORT variant
# ---------------------------------------------------------------------------


class TestCTMSORT:
    def _make_sort_model(self, **kw):
        from baseline.models.ctm_sort import ContinuousThoughtMachineSORT
        params = dict(
            iterations=4,
            d_model=32,
            d_input=8,
            heads=0,
            n_synch_out=4,
            n_synch_action=0,
            synapse_depth=1,
            memory_length=5,
            deep_nlms=True,
            memory_hidden_dims=2,
            do_layernorm_nlm=False,
            backbone_type="none",
            positional_embedding_type="none",
            out_dims=10,
            prediction_reshaper=[-1],
            dropout=0.0,
            neuron_select_type="random-pairing",
            n_random_pairing_self=0,
        )
        params.update(kw)
        return ContinuousThoughtMachineSORT(**params)

    def test_construction(self):
        model = self._make_sort_model()
        assert model.attention is None
        assert model.q_proj is None

    def test_forward_shape(self):
        model = self._make_sort_model()
        B, D = 2, 8
        x = torch.randn(B, D)
        preds, certs, synch = model(x)
        assert preds.shape == (B, 10, model.iterations)
        assert certs.shape == (B, 2, model.iterations)

    def test_no_nans(self):
        model = self._make_sort_model()
        x = torch.randn(2, 8)
        preds, _, _ = model(x)
        assert not torch.isnan(preds).any()

    def test_gradient_flows(self):
        model = self._make_sort_model()
        x = torch.randn(2, 8)
        preds, _, _ = model(x)
        preds.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# RL variant
# ---------------------------------------------------------------------------


class TestCTMRL:
    def _make_rl_model(self, **kw):
        from baseline.models.ctm_rl import ContinuousThoughtMachineRL
        params = dict(
            iterations=3,
            d_model=32,
            d_input=4,
            n_synch_out=4,
            synapse_depth=1,
            memory_length=5,
            deep_nlms=True,
            memory_hidden_dims=2,
            do_layernorm_nlm=False,
            backbone_type="classic-control-backbone",
            dropout=0.0,
            neuron_select_type="first-last",
        )
        params.update(kw)
        return ContinuousThoughtMachineRL(**params)

    def test_construction(self):
        model = self._make_rl_model()
        assert model.attention is None
        assert model.output_projector is None

    def test_forward_shape(self):
        model = self._make_rl_model()
        B = 2
        x = torch.randn(B, 4)
        state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
        act_trace = model.start_activated_trace.unsqueeze(0).expand(B, -1, -1)
        synch, (new_st, new_at) = model(x, (state_trace, act_trace))
        n = model.n_synch_out
        expected = n * (n + 1) // 2
        assert synch.shape == (B, expected)
        assert new_st.shape == state_trace.shape
        assert new_at.shape == act_trace.shape

    def test_no_nans(self):
        model = self._make_rl_model()
        B = 1
        x = torch.randn(B, 4)
        state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
        act_trace = model.start_activated_trace.unsqueeze(0).expand(B, -1, -1)
        synch, _ = model(x, (state_trace, act_trace))
        assert not torch.isnan(synch).any()

    def test_forward_with_tracking(self):
        model = self._make_rl_model()
        model.eval()
        B = 1
        x = torch.randn(B, 4)
        state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
        act_trace = model.start_activated_trace.unsqueeze(0).expand(B, -1, -1)
        result = model(x, (state_trace, act_trace), track=True)
        assert len(result) == 4
        synch, hidden, pre_act, post_act = result
        assert pre_act.shape[0] == model.iterations
        assert post_act.shape[0] == model.iterations

    def test_gradient_flows(self):
        model = self._make_rl_model()
        B = 2
        x = torch.randn(B, 4)
        state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
        act_trace = model.start_activated_trace.unsqueeze(0).expand(B, -1, -1)
        synch, _ = model(x, (state_trace, act_trace))
        synch.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------


class TestLSTMBaseline:
    def _make_lstm(self, **kw):
        from baseline.models.lstm import LSTMBaseline
        params = dict(
            iterations=4,
            d_model=32,
            d_input=8,
            heads=2,
            backbone_type="parity_backbone",
            positional_embedding_type="custom-rotational-1d",
            num_layers=1,
            out_dims=16,
            prediction_reshaper=[-1],
            dropout=0.0,
        )
        params.update(kw)
        return LSTMBaseline(**params)

    def test_parity_forward(self):
        model = self._make_lstm()
        parity_length = 8
        x = torch.randint(0, 2, (2, parity_length)).float() * 2 - 1
        preds, certs, _ = model(x)
        assert preds.shape == (2, 16, model.iterations)
        assert certs.shape == (2, 2, model.iterations)

    def test_no_nans(self):
        model = self._make_lstm()
        x = torch.randint(0, 2, (1, 8)).float() * 2 - 1
        preds, _, _ = model(x)
        assert not torch.isnan(preds).any()

    def test_gradient_flows(self):
        model = self._make_lstm()
        x = torch.randint(0, 2, (1, 8)).float() * 2 - 1
        preds, _, _ = model(x)
        preds.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# FF baseline
# ---------------------------------------------------------------------------


class TestFFBaseline:
    def test_resnet18_construction(self):
        from baseline.models.ff import FFBaseline
        model = FFBaseline(
            d_model=32,
            backbone_type="resnet18-1",
            out_dims=10,
        )
        assert model is not None

    def test_forward_shape(self):
        from baseline.models.ff import FFBaseline
        model = FFBaseline(
            d_model=32,
            backbone_type="resnet18-1",
            out_dims=10,
        )
        x = torch.randn(2, 3, 28, 28)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[-1] == 10

    def test_no_nans(self):
        from baseline.models.ff import FFBaseline
        model = FFBaseline(
            d_model=32,
            backbone_type="resnet18-1",
            out_dims=10,
        )
        x = torch.randn(1, 3, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Synchronisation mechanics
# ---------------------------------------------------------------------------


class TestSynchronisationMechanics:
    def test_certainty_sums_to_one(self):
        parity_length = 8
        params = _base_params(
            backbone_type="parity_backbone",
            positional_embedding_type="custom-rotational-1d",
            out_dims=2 * parity_length,
            prediction_reshaper=[parity_length, 2],
            d_input=8,
        )
        model = ContinuousThoughtMachine(**params)
        x = torch.randint(0, 2, (1, parity_length)).float() * 2 - 1
        preds, certs, _ = model(x)
        cert_sums = certs.sum(dim=1)
        assert torch.allclose(cert_sums, torch.ones_like(cert_sums), atol=1e-5)

    @pytest.mark.parametrize("synch_type", ["out", "action"])
    def test_decay_params_shape(self, synch_type):
        model = ContinuousThoughtMachine(**_base_params())
        param = getattr(model, f"decay_params_{synch_type}")
        rep_size = getattr(model, f"synch_representation_size_{synch_type}")
        assert param.shape == (rep_size,)

    def test_neuron_indices_within_d_model(self):
        model = ContinuousThoughtMachine(**_base_params())
        for synch_type in ["out", "action"]:
            left = getattr(model, f"{synch_type}_neuron_indices_left")
            right = getattr(model, f"{synch_type}_neuron_indices_right")
            assert left.max() < model.d_model
            assert right.max() < model.d_model


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_neuron_select_types(self):
        assert set(VALID_NEURON_SELECT_TYPES) == {"first-last", "random", "random-pairing"}

    def test_valid_backbone_types_includes_parity(self):
        assert "parity_backbone" in VALID_BACKBONE_TYPES
        assert "shallow-wide" in VALID_BACKBONE_TYPES

    def test_valid_backbone_types_resnet_variants(self):
        for depth in [18, 34, 50, 101, 152]:
            for i in range(1, 5):
                assert f"resnet{depth}-{i}" in VALID_BACKBONE_TYPES

    def test_valid_positional_embedding_types(self):
        assert "learnable-fourier" in VALID_POSITIONAL_EMBEDDING_TYPES
        assert "custom-rotational" in VALID_POSITIONAL_EMBEDDING_TYPES
        assert "custom-rotational-1d" in VALID_POSITIONAL_EMBEDDING_TYPES
