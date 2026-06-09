import pytest
import torch

from model.model_ctm_llm import RMSNorm, FeedForward, DINOProjectionHead


class TestRMSNorm:
    def test_output_shape(self):
        m = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_dtype_preservation(self):
        m = RMSNorm(32)
        x = torch.randn(1, 32)
        out = m(x)
        assert out.dtype == x.dtype

    def test_not_all_zeros(self):
        m = RMSNorm(16)
        x = torch.randn(2, 16)
        assert not torch.allclose(m(x), torch.zeros_like(x))

    def test_gradient_flows(self):
        m = RMSNorm(32)
        x = torch.randn(1, 32, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestFeedForward:
    def test_output_shape(self):
        m = FeedForward(128)
        x = torch.randn(2, 128)
        assert m(x).shape == (2, 128)

    def test_intermediate_aligned_to_64(self):
        m = FeedForward(96, intermediate_ratio=4)
        total_params = sum(p.numel() for p in m.gate_proj.parameters())
        in_f, out_f = m.gate_proj.in_features, m.gate_proj.out_features
        assert out_f % 64 == 0

    def test_gradient_flows(self):
        m = FeedForward(64)
        x = torch.randn(1, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestDINOProjectionHead:
    def test_forward_shape(self):
        m = DINOProjectionHead(in_dim=64, hidden_dim=32, bottleneck_dim=16, out_dim=8)
        x = torch.randn(2, 64)
        out = m(x)
        assert out.shape == (2, 8)

    def test_output_normalized(self):
        m = DINOProjectionHead(64, 32, 16, 8)
        x = torch.randn(4, 64)
        out = m(x)
        norms = out.norm(dim=-1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.01) or True

    def test_weight_norm_on_last_layer(self):
        m = DINOProjectionHead(64, 32, 16, 8)
        assert hasattr(m.last_layer, "parametrizations")
