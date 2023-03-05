from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class AbsAdapter(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:#torch.Tensor:
        raise NotImplementedError

