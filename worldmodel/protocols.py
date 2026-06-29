"""Protocols (structural typing) for world models, solvers, and transforms.

These mirror the lightweight Actionable / Costable contracts from
stable-worldmodel: a world model only has to implement ``get_cost`` to be
planned against by a sampling solver such as CEM, and optionally
``get_action`` for warm-starting / behavioural-cloning policies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Costable(Protocol):
    """Anything that can score action candidates against a goal state.

    Solvers (CEM, MPPI, ...) call ``get_cost`` with an info dict describing
    the current state + goal, plus a batch of candidate action sequences, and
    expect back one cost per candidate (lower == better).
    """

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Return cost per candidate, shape ``(n_envs, num_samples)``.

        Args:
            info_dict: state + goal. Conventionally carries an observation key
                (``'pixels'`` for images or ``'state'`` for vectors) plus a
                ``'goal'`` key, broadcast to ``(n_envs, num_samples, ...)``.
            action_candidates: ``(n_envs, num_samples, horizon, action_dim)``.
        """
        ...


@runtime_checkable
class Actionable(Protocol):
    """Anything that can directly propose an action from an observation.

    Used by FeedForwardPolicy (behavioural cloning) and to warm-start solvers.
    """

    def get_action(
        self,
        info: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


class Transformable(Protocol):
    """Reversible preprocessing (normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray: ...

    def inverse_transform(self, x: np.ndarray) -> np.ndarray: ...
