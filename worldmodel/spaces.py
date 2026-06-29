"""Minimal space helpers (no gymnasium dependency).

A ``Box`` is a contiguous real-valued space with ``low`` / ``high`` arrays and
a ``shape``. It is all a solver needs to sample / clip actions, and all an env
needs to advertise its observation / action spaces. A small running-statistics
normalizer (``RunningStats``) implements the ``Transformable`` protocol for
optional observation / action standardization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    """A bounded continuous space.

    Attributes:
        low:  array (broadcastable to ``shape``) of lower bounds.
        high: array of upper bounds.
        shape: sample shape (without a leading batch dim).
        dtype: numpy dtype of samples.
    """

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: type = np.float32

    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        low = np.broadcast_to(np.asarray(low, dtype=self.dtype), np.asarray(shape) if shape is not None else np.asarray(np.shape(low)))
        high = np.broadcast_to(np.asarray(high, dtype=self.dtype), low.shape)
        self.low = low
        self.high = high
        self.shape = tuple(low.shape)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(
            np.all(x >= self.low) and np.all(x <= self.high)
        )

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.low, self.high)


class RunningStats:
    """Welford running mean/std implementing the Transformable protocol."""

    def __init__(self, shape=(), eps: float = 1e-6):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps  # avoid div-by-zero

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.reshape(-1, *self.mean.shape).mean(axis=0)
        batch_var = x.reshape(-1, *self.mean.shape).var(axis=0)
        batch_count = x.reshape(-1, *self.mean.shape).shape[0]
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (self.std + 1e-8)).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * (self.std + 1e-8) + self.mean).astype(np.float32)
