import torch
import torch_npu


def _dsv4_clamped_swiglu_reference(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return torch_npu.npu_swiglu(torch.cat((gate, up), dim=-1))


@torch.inference_mode()
def test_clipped_swiglu_matches_dsv4_prefill_activation() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu") * 4

    expected = _dsv4_clamped_swiglu_reference(x, limit=10.0)
    actual = torch_npu.npu_clipped_swiglu(
        x,
        interleaved=False,
        alpha=1.0,
        limit=10.0,
        bias=0.0,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_quant, expected_scale = torch_npu.npu_dynamic_quant(expected)
    actual_quant, actual_scale = torch_npu.npu_dynamic_quant(actual)
    torch.testing.assert_close(actual_quant, expected_quant, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
