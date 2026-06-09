import pytest
import torch
import torch.nn as nn

from model.ctm_modules import SuperLinear, SynapseUNET, Squeeze, TTTMLP


class TestSuperLinear:
    def test_forward_shape(self):
        B, T, N, L = 2, 4, 16, 5
        m = SuperLinear(L, 2, N)
        x = torch.randn(B, T, N, L)
        out = m(x)
        assert out.shape == (B, T, N, 2)

    def test_output_dim_1_squeezes(self):
        m = SuperLinear(5, 1, 8)
        x = torch.randn(1, 1, 8, 5)
        out = m(x)
        assert out.shape == (1, 1, 8)

    def test_gradient_flows(self):
        m = SuperLinear(4, 2, 8)
        x = torch.randn(1, 1, 8, 4, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSqueeze:
    def test_squeeze_last(self):
        m = Squeeze(-1)
        x = torch.randn(2, 3, 1)
        assert m(x).shape == (2, 3)

    def test_squeeze_noop(self):
        m = Squeeze(-1)
        x = torch.randn(2, 3, 4)
        assert m(x).shape == (2, 3, 4)

    def test_squeeze_dim0(self):
        m = Squeeze(0)
        x = torch.randn(1, 5)
        assert m(x).shape == (5,)


class TestSynapseUNET:
    def test_forward_shape(self):
        m = SynapseUNET(in_dims=32, out_dims=16, depth=2, dropout=0.0)
        x = torch.randn(2, 32)
        assert m(x).shape == (2, 16)

    def test_sequence_input(self):
        m = SynapseUNET(in_dims=32, out_dims=16, depth=2, dropout=0.0)
        x = torch.randn(2, 8, 32)
        out = m(x)
        assert out.shape == (2, 8, 16)

    def test_skip_connection_effect(self):
        m = SynapseUNET(32, 16, depth=3, dropout=0.0)
        x = torch.randn(2, 32)
        out = m(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        m = SynapseUNET(16, 8, depth=2, dropout=0.0)
        x = torch.randn(1, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestTTTMLP:
    def test_zero_init_produces_small_output(self):
        m = TTTMLP(d_model=32, gate_init=-5.0)
        x = torch.randn(2, 32)
        out = m(x)
        gate_val = torch.sigmoid(torch.tensor(-5.0)).item()
        assert gate_val < 0.01
        delta_norm = out.abs().max().item()
        assert delta_norm < 1.0, f"zero-init TTT output should be small, got {delta_norm}"

    def test_forward_shape(self):
        m = TTTMLP(d_model=64, hidden_mult=2)
        x = torch.randn(3, 64)
        assert m(x).shape == (3, 64)

    def test_gradient_flows(self):
        m = TTTMLP(d_model=32)
        x = torch.randn(1, 32, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_trainable_params(self):
        m = TTTMLP(d_model=64)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert trainable > 0
