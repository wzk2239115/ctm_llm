"""Episode replay buffer and a sequence-chunk dataset for JEPA training.

The buffer stores whole episodes (dicts of per-step arrays). The dataset
samples fixed-length chunks and yields ``(H+1)`` frames plus the ``H``
inter-frame actions, which is exactly what an action-conditioned latent
predictor needs: encode frame ``t``, roll out ``H`` actions, regress the
predicted latents against the (stop-gradiented) encoder latents of frames
``t+1 .. t+H``.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Append-only store of variable-length episodes.

    Episodes are dicts of arrays sharing a leading time dimension. The
    observation modality is discovered automatically: a buffer that received
    ``'pixels'`` exposes ``obs_key == 'pixels'``; one that received ``'state'``
    exposes ``obs_key == 'state'``.
    """

    def __init__(self):
        self.episodes: list[dict[str, np.ndarray]] = []
        self._current: dict[str, list] | None = None

    @property
    def obs_key(self) -> str:
        if not self.episodes:
            raise RuntimeError('ReplayBuffer is empty; obs_key unknown.')
        cols = self.episodes[0].keys()
        return 'pixels' if 'pixels' in cols else 'state'

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0]['action'].shape[1])

    @property
    def total_steps(self) -> int:
        return sum(ep['action'].shape[0] for ep in self.episodes)

    def begin_episode(self) -> None:
        self._current = {}

    def add(self, obs_dict: dict, action: np.ndarray) -> None:
        if self._current is None:
            self.begin_episode()
        for k, v in obs_dict.items():
            self._current.setdefault(k, []).append(np.asarray(v))
        self._current.setdefault('action', []).append(np.asarray(action))

    def end_episode(self) -> dict[str, np.ndarray]:
        if self._current is None:
            raise RuntimeError('No episode in progress.')
        ep = {k: np.stack(v, axis=0) for k, v in self._current.items()}
        self.episodes.append(ep)
        self._current = None
        return ep

    def add_episode(self, ep: dict[str, np.ndarray]) -> None:
        self.episodes.append({k: np.asarray(v) for k, v in ep.items()})

    def obs_array(self) -> np.ndarray:
        """Concatenate the observation column across all episodes."""
        key = self.obs_key
        return np.concatenate([ep[key] for ep in self.episodes], axis=0)


class ChunkDataset(torch.utils.data.Dataset):
    """Yields ``(H+1)``-frame chunks with the matching ``H`` actions.

    Args:
        buffer: a filled :class:`ReplayBuffer`.
        horizon: number of predicted transitions ``H`` (chunk has ``H+1``
            frames).
        frameskip: stride between consecutive frames in a chunk.
        transform: optional dict-in / dict-out transform (e.g. normalization).
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        horizon: int,
        frameskip: int = 1,
        transform=None,
    ):
        self.buffer = buffer
        self.horizon = int(horizon)
        self.frameskip = int(frameskip)
        self.span = (self.horizon + 1) * self.frameskip
        self.transform = transform
        self.obs_key = buffer.obs_key
        self._clips: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(buffer.episodes):
            length = ep['action'].shape[0]  # transitions = frames - 1
            n_frames = length + 1
            if n_frames >= self.span:
                for start in range(0, n_frames - self.span + 1):
                    self._clips.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._clips)

    def _slice_frames(self, ep: dict, start: int) -> tuple[np.ndarray, np.ndarray]:
        idx = [start + i * self.frameskip for i in range(self.horizon + 1)]
        obs = np.stack([ep[self.obs_key][j] for j in idx], axis=0)  # (H+1, ...)
        # actions aligned with transitions: action[t] leads frame[t] -> frame[t+1]
        act_idx = [start + i * self.frameskip for i in range(self.horizon)]
        act = np.stack([ep['action'][j] for j in act_idx], axis=0)  # (H, A)
        return obs, act

    def __getitem__(self, i: int) -> dict:
        ep_idx, start = self._clips[i]
        ep = self.buffer.episodes[ep_idx]
        obs, act = self._slice_frames(ep, start)
        out = {self.obs_key: obs, 'action': act}
        if self.transform is not None:
            out = self.transform(out)
        return out


def collate(batch: list[dict]) -> dict:
    """Stack a list of sample dicts into batched tensors."""
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        out[k] = torch.as_tensor(np.stack([b[k] for b in batch], axis=0))
    return out
