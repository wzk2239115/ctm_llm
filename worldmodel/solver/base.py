"""Solver protocol and base helpers.

A solver plans an action sequence by querying a :class:`~worldmodel.protocols.Costable`
world model. It is configured once per environment (``configure``) and called
per (re)plan with the current info dict (already sliced to the envs that need
a new plan) plus an optional warm-start action sequence.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from worldmodel.spaces import Box


@runtime_checkable
class Solver(Protocol):
    def configure(
        self, *, action_space: Box, n_envs: int, config: Any
    ) -> None: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def horizon(self) -> int: ...

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict: ...


def _box_bounds(action_space: Box, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.as_tensor(np.array(action_space.low), dtype=dtype, device=device)
    high = torch.as_tensor(np.array(action_space.high), dtype=dtype, device=device)
    return low, high
