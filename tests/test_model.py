import numpy as np
import torch
from rmi.model.plu import PLU


def test_plu_activation():
    input_sample = torch.Tensor(np.linspace(-2, 2, num=41))
    out = PLU(input_sample)
    assert out[0] == 0.1 * (-2 + 1) - 1
    assert out[-1] == 0.1 * (2 - 1) + 1
