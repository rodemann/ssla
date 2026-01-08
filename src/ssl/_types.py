from typing import TypedDict

import torch


class SSLAResult(TypedDict):
    y: torch.Tensor
    log_ppd: torch.Tensor
