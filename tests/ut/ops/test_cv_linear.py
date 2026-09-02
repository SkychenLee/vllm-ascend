from unittest.mock import MagicMock

import torch

from vllm_ascend.ops.cv_linear import CVLinearWrapper


def test_forward_only_wrapper_is_not_split() -> None:
    linear = MagicMock(spec=["forward"])
    linear.forward.side_effect = lambda x: x + 1
    wrapper = CVLinearWrapper(linear)
    x = torch.zeros(2, 4)

    quantized, scale = wrapper.quantize(x)
    output = wrapper.matmul(quantized, scale)

    assert quantized is x
    assert scale is None
    assert torch.equal(output, x + 1)
    linear.forward.assert_called_once_with(x)
