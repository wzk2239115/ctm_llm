"""External-env adapters: plug DMC locomotion / Atari / Craftax into the
worldmodel goal-conditioned Env protocol (self-rolled, no stable-worldmodel /
ogbench).

These wrap third-party envs (which follow the gymnasium-style API:
``reset() -> obs``, ``step(action) -> (obs, reward, term, trunc, info)``) into
our dict protocol::

    obs = env.reset(seed)          # {'state'/'pixels': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

Dependencies (mujoco / dm_control / gymnasium / ale-py / craftax / jax) are
imported lazily *inside* the factory functions, so importing this module never
fails even when none are installed. Each factory raises ``ImportError`` with the
exact ``pip install ...`` command when its dependency is missing.

Goal synthesis
--------------
DMC / Atari / Craftax are reward-maximization, *not* goal-conditioned, so we
synthesize a goal-conditioned task on top:

* On ``reset`` we sample a goal observation (a Gaussian perturbation of the
  initial obs for state envs, a copy of the initial frame for image envs).
* ``success`` = (``L2(obs, goal) < thresh``) OR native termination, with an
  optional reward-based fallback (cumulative reward >= ``reward_thresh``) for
  envs where the distance-to-goal semantics are unnatural (e.g. Atari).
* ``reward = float(success)`` (sparse), ``truncated = step >= max_steps``.

Public API:
    GymAdapter              — wrap any gymnasium-API env into our protocol
    make_dmc(domain, task)  — dm_control.suite locomotion (state by default)
    make_atari(game)        — gymnasium ALE (image, grayscale 64x64)
    make_craftax(variant)   — craftax classic/full (symbolic or pixels)
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


# ============================================================
# space / obs conversion helpers
# ============================================================

def _wm_box_from_gym(space):
    """Convert a gymnasium (or old gym) space into ``worldmodel.spaces.Box``.

    Handles ``Box`` / ``Discrete`` / ``MultiBinary`` / ``MultiDiscrete`` and
    passes our own ``Box`` through unchanged. Falls back to duck-typing on a
    space with ``low`` / ``high`` / ``shape`` attributes.
    """
    if isinstance(space, Box):
        return space
    cls_name = type(space).__name__
    # Box
    if cls_name == 'Box' and hasattr(space, 'low') and hasattr(space, 'shape'):
        return Box(low=np.asarray(space.low, dtype=np.float32),
                   high=np.asarray(space.high, dtype=np.float32),
                   shape=tuple(space.shape))
    # Discrete -> scalar index box [0, n-1]
    if cls_name == 'Discrete':
        n = int(getattr(space, 'n', 2))
        return Box(low=0.0, high=float(n - 1), shape=(1,))
    # MultiBinary
    if cls_name == 'MultiBinary':
        n = int(getattr(space, 'n', np.asarray(getattr(space, 'shape', [2])).prod()))
        return Box(low=0.0, high=1.0, shape=(n,))
    # MultiDiscrete
    if cls_name == 'MultiDiscrete':
        nvec = np.asarray(getattr(space, 'nvec'), dtype=np.float32)
        return Box(low=np.zeros_like(nvec), high=nvec - 1.0, shape=nvec.shape)
    # duck-typed Box fallback
    if hasattr(space, 'low') and hasattr(space, 'high') and hasattr(space, 'shape'):
        return Box(low=np.asarray(space.low, dtype=np.float32),
                   high=np.asarray(space.high, dtype=np.float32),
                   shape=tuple(space.shape))
    raise TypeError(f"Cannot convert gym space {space!r} to worldmodel Box")


def _action_kind(space):
    cls_name = type(space).__name__
    if cls_name == 'Discrete':
        return ('discrete', int(getattr(space, 'n', 2)))
    if cls_name == 'MultiDiscrete':
        return ('multidiscrete', np.asarray(getattr(space, 'nvec')))
    if cls_name == 'MultiBinary':
        return ('multibinary', int(getattr(space, 'n', np.asarray(getattr(space, 'shape', [2])).prod())))
    return ('box', None)


def _encode_action(kind, action):
    """Map a worldmodel action array back to the underlying env's action type."""
    name, meta = kind
    a = np.asarray(action, dtype=np.float32).flatten()
    if name == 'discrete':
        idx = int(np.clip(round(a[0]), 0, meta - 1)) if a.size else 0
        return idx
    if name == 'multidiscrete':
        return np.clip(np.round(a).astype(np.int64), 0, meta - 1)
    if name == 'multibinary':
        return (a >= 0.5).astype(np.int64)
    return a


def _to_chw_float01(img):
    """Coerce a raw image obs (HWC uint8 / HWC float / HW) to (C,H,W) float in [0,1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        # assume HWC when last axis is small (<=4) and differs from first
        if img.shape[-1] <= 4 and img.shape[-1] != img.shape[0]:
            img = np.transpose(img, (2, 0, 1))
    return np.clip(img, 0.0, 1.0)


def _resize_nn(img_hw_or_hwc, size):
    """Nearest-neighbour downsample to (size, size). Works for 2D or 3D (HWC)."""
    arr = np.asarray(img_hw_or_hwc)
    h, w = arr.shape[:2]
    ys = np.clip((np.arange(size, dtype=np.float32) * h / size).astype(np.int64), 0, h - 1)
    xs = np.clip((np.arange(size, dtype=np.float32) * w / size).astype(np.int64), 0, w - 1)
    return arr[np.ix_(ys, xs)] if arr.ndim == 2 else arr[np.ix_(ys, xs)]


def _center_crop_square(img_hwc):
    arr = np.asarray(img_hwc)
    h, w = arr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return arr[y0:y0 + s, x0:x0 + s]


# ============================================================
# GymAdapter — the core wrapper
# ============================================================

class GymAdapter:
    """Wrap a gymnasium-API env into the worldmodel goal-conditioned protocol.

    The wrapped ``gym_env`` must expose::

        reset(seed=None) -> obs  | (obs, info)
        step(action)     -> (obs, reward, term, trunc, info)   # 5-tuple
                         |  (obs, reward, term, info)          # 4-tuple (old gym)
        observation_space  (gymnasium Box / Discrete / ...)
        action_space       (gymnasium Box / Discrete / ...)

    Args:
        gym_env: the underlying env following the gymnasium API above.
        image: if True, obs is an image and obs_dict uses the 'pixels' key
            (both obs and goal); else obs is a flat vector ('state' key).
        goal_thresh: L2 threshold for goal-reaching success (None -> heuristic).
        goal_noise: std of the Gaussian perturbation sampling the goal obs
            (None -> 0.2 * median finite span). Ignored when goal_mode='fixed'.
        goal_mode: 'perturb' (goal = obs + noise, default for state) or 'fixed'
            (goal = initial obs, default for image).
        success_mode: 'goal' (L2 < thresh), 'native' (underlying term flag), or
            'reward' (cumulative episode reward >= reward_thresh).
        reward_thresh: threshold for success_mode='reward'.
        max_steps: truncation horizon (None -> env.spec.max_episode_steps or 200).
        seed: RNG seed for goal sampling.
    """

    def __init__(self, gym_env, image=False, goal_thresh=None, goal_noise=None,
                 goal_mode=None, success_mode='goal', reward_thresh=1.0,
                 max_steps=None, seed=None):
        self.gym = gym_env
        self.image = bool(image)
        self.rng = np.random.default_rng(seed)

        self._obs_space = _wm_box_from_gym(gym_env.observation_space)
        self._act_space = _wm_box_from_gym(gym_env.action_space)
        self._act_kind = _action_kind(gym_env.action_space)
        self._goal_space = self._obs_space  # goal is a target observation

        # heuristic goal noise / thresh from finite spans of the obs space
        low = self._obs_space.low
        high = self._obs_space.high
        finite = np.isfinite(low) & np.isfinite(high)
        spans = np.where(finite, high - low, 1.0).flatten()
        spans = spans[spans > 0] if np.any(spans > 0) else np.array([1.0])
        median_span = float(np.median(spans))

        self.goal_mode = goal_mode or ('fixed' if self.image else 'perturb')
        self.success_mode = success_mode
        self.reward_thresh = float(reward_thresh)
        self.goal_noise = float(goal_noise) if goal_noise is not None else 0.2 * median_span
        if goal_thresh is not None:
            self.goal_thresh = float(goal_thresh)
        elif self.image:
            self.goal_thresh = 0.10 * float(np.sqrt(np.prod(self._obs_space.shape)))
        else:
            self.goal_thresh = 0.15 * median_span * float(np.sqrt(self._obs_space.shape[0]))

        ms = max_steps
        if ms is None:
            spec = getattr(gym_env, 'spec', None)
            ms = getattr(spec, 'max_episode_steps', None) if spec is not None else None
        self.max_steps = int(ms) if ms is not None else 200

        self._step = 0
        self._cur_obs = None
        self._goal_obs = None
        self._cum_reward = 0.0

    # ---- spaces ----
    @property
    def observation_space(self):
        return self._obs_space

    @property
    def goal_space(self):
        return self._goal_space

    @property
    def action_space(self):
        return self._act_space

    # ---- goal sampling ----
    def _sample_goal(self, obs):
        if self.goal_mode == 'fixed':
            return obs.astype(np.float32).copy()
        # 'perturb'
        noise = self.rng.normal(0.0, self.goal_noise, size=obs.shape).astype(np.float32)
        goal = obs + noise
        # clip only to finite bounds
        low = np.where(np.isfinite(self._obs_space.low), self._obs_space.low, goal)
        high = np.where(np.isfinite(self._obs_space.high), self._obs_space.high, goal)
        return np.clip(goal, low, high).astype(np.float32)

    # ---- protocol ----
    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            try:
                result = self.gym.reset(seed=int(seed))
            except TypeError:
                result = self.gym.reset()
        else:
            result = self.gym.reset()
        obs = self._unpack_reset(result)
        obs = self._normalize_obs(obs)
        self._cur_obs = obs
        if goal is not None:
            g = self._normalize_obs(goal)
        else:
            g = self._sample_goal(obs)
        self._goal_obs = g
        self._step = 0
        self._cum_reward = 0.0
        return self._wrap_obs()

    def step(self, action):
        gaction = _encode_action(self._act_kind, action)
        out = self.gym.step(gaction)
        obs, reward, term, trunc, info = self._unpack_step(out)
        obs = self._normalize_obs(obs)
        self._cur_obs = obs
        self._step += 1
        self._cum_reward += float(reward)

        dist = float(np.linalg.norm(obs - self._goal_obs))
        if self.success_mode == 'reward':
            success = self._cum_reward >= self.reward_thresh
        elif self.success_mode == 'native':
            success = bool(term)
        else:  # 'goal'
            success = (dist < self.goal_thresh) or bool(term)

        terminated = bool(success)
        truncated = bool(trunc or self._step >= self.max_steps)
        info = dict(info) if isinstance(info, dict) else {}
        info.update({'distance': dist, 'native_term': bool(term),
                     'native_reward': float(reward),
                     'cumulative_reward': self._cum_reward})
        return self._wrap_obs(), float(terminated), terminated, truncated, info

    # ---- helpers ----
    @staticmethod
    def _unpack_reset(result):
        # gymnasium: (obs, info); old gym / our adapters: obs
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0]
        return result

    @staticmethod
    def _unpack_step(out):
        if len(out) == 5:
            obs, reward, term, trunc, info = out
        elif len(out) == 4:
            obs, reward, term, info = out
            trunc = False
        else:
            raise RuntimeError(f"Unexpected step() return arity {len(out)}")
        return obs, reward, term, trunc, info

    def _normalize_obs(self, obs):
        obs = np.asarray(obs)
        if self.image:
            return _to_chw_float01(obs)
        return obs.astype(np.float32).flatten()

    def _wrap_obs(self):
        key = 'pixels' if self.image else 'state'
        return {key: self._cur_obs.copy(), 'goal': self._goal_obs.copy()}


# ============================================================
# DMC locomotion (dm_control.suite) — state obs (flatten) by default
# ============================================================

class _DMCToGym:
    """Adapt dm_control.suite env (Timestep API) to the gymnasium reset/step API."""

    def __init__(self, dmc_env, image=False, image_size=64, camera_id=-1):
        self.dmc = dmc_env
        self.image = bool(image)
        self.image_size = int(image_size)
        self.camera_id = camera_id

        asp = dmc_env.action_spec()
        self.action_space = Box(low=np.asarray(asp.minimum, dtype=np.float32),
                                high=np.asarray(asp.maximum, dtype=np.float32),
                                shape=tuple(asp.shape))

        if self.image:
            self.observation_space = Box(0.0, 1.0, (3, self.image_size, self.image_size))
        else:
            spec = dmc_env.observation_spec()
            self._obs_keys = list(spec.keys())
            total = int(sum(np.prod(np.asarray(spec[k].shape, dtype=np.int64)) for k in self._obs_keys))
            self.observation_space = Box(-np.inf, np.inf, (total,))
        self._ts = None

    def reset(self, seed=None):
        self._ts = self.dmc.reset()
        return self._obs()

    def step(self, action):
        self._ts = self.dmc.step(np.asarray(action, dtype=np.float64))
        obs = self._obs()
        reward = 0.0 if self._ts.reward is None else float(self._ts.reward)
        term = self._ts.last()
        discount = 0.0 if self._ts.discount is None else float(self._ts.discount)
        return obs, reward, term, False, {'discount': discount}

    def _obs(self):
        if self.image:
            img = self.dmc.physics.render(self.image_size, self.image_size,
                                          camera_id=self.camera_id)
            return np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        parts = [np.asarray(self._ts.observation[k], dtype=np.float32).flatten()
                 for k in self._obs_keys]
        return np.concatenate(parts).astype(np.float32)


def make_dmc(domain, task='run', seed=None, image=False, image_size=64,
             camera_id=-1, **adapter_kwargs):
    """Build a goal-conditioned wrapper around a dm_control.suite locomotion env.

    Examples: cheetah:run, walker:walk, hopper:hop, acrobot:swingup,
    reacher:easy, pendulum, cartpole:balance, humanoid:walk, finger:spin,
    quadruped:walk. State obs = flattened dict of arrays; set ``image=True`` for
    rendered pixels (needs an EGL/display context).
    """
    try:
        from dm_control import suite
    except ImportError as e:
        raise ImportError(
            "dm_control is required for DMC envs.\n"
            "  pip install dm_control mujoco\n"
            "(image mode additionally needs an OpenGL context: set EGL_PLATFORM=true / "
            "MUJOCO_GL=egl for headless rendering)"
        ) from e
    env = suite.load(domain_name=domain, task_name=task,
                     task_kwargs={'random': int(seed) if seed is not None else 0})
    gym_env = _DMCToGym(env, image=image, image_size=image_size, camera_id=camera_id)
    adapter_kwargs.setdefault('success_mode', 'native')
    return GymAdapter(gym_env, image=image, seed=seed, **adapter_kwargs)


# ============================================================
# Atari (gymnasium + ale-py) — image obs, grayscale 64x64
# ============================================================

class _AtariPreprocess:
    """Resize + grayscale Atari frames to (C, 64, 64) float in [0,1]."""

    def __init__(self, env, size=64, gray=True):
        self.env = env
        self.size = int(size)
        self.gray = bool(gray)
        self.action_space = env.action_space
        self._channels = 1 if self.gray else 3
        self.observation_space = Box(0.0, 1.0, (self._channels, self.size, self.size))

    def _proc(self, obs):
        img = np.asarray(obs)
        if img.ndim == 2:
            return _resize_nn(img, self.size).astype(np.float32)[None, ...] / 255.0
        img = _center_crop_square(img)  # (H,W,3) uint8
        if self.gray:
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
            img = _resize_nn(img, self.size).astype(np.float32)[None, ...] / 255.0
        else:
            img = _resize_nn(img, self.size).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
        return np.clip(img / 255.0, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset(seed=seed)
        obs = result[0] if isinstance(result, tuple) else result
        return self._proc(obs)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, r, term, trunc, info = out
        else:
            obs, r, term, info = out
            trunc = False
        return self._proc(obs), r, term, trunc, info


def make_atari(game, seed=None, image_size=64, gray=True, frameskip=4,
               **adapter_kwargs):
    """Build a goal-conditioned wrapper around an Atari game via gymnasium ALE.

    ``game`` is the bare name (e.g. 'Pong', 'Breakout'); resolves to ``ALE/{game}-v5``.
    Image obs = grayscale (or RGB) 64x64. Atari is reward-max, so the default
    success_mode is 'reward' (cumulative reward >= reward_thresh; default 1.0 —
    override per game). The goal frame is a copy of the initial frame (trivially
    "reached") so the protocol is uniform; success is driven by reward.
    """
    try:
        import gymnasium as gym
    except ImportError as e:
        raise ImportError(
            "gymnasium + ale-py are required for Atari envs.\n"
            "  pip install gymnasium ale-py\n"
            "  AutoROM -v  # downloads the Atari ROMs"
        ) from e
    env = gym.make(f"ALE/{game}-v5",
                   frameskip=frameskip, repeat_action_probability=0.0,
                   obs_type='rgb')
    env = _AtariPreprocess(env, size=image_size, gray=gray)
    adapter_kwargs.setdefault('success_mode', 'reward')
    adapter_kwargs.setdefault('goal_mode', 'fixed')
    adapter_kwargs.setdefault('reward_thresh', 1.0)
    return GymAdapter(env, image=True, seed=seed, **adapter_kwargs)


# ============================================================
# Craftax (jax) — symbolic or pixels
# ============================================================

class _CraftaxToGym:
    """Adapt craftax's functional jax API to the gymnasium reset/step API.

    Craftax envs are flax structs with a functional step:
        state, obs = env.reset(key)
        state, obs, reward, done, info = env.step(key, state, action)
    Symbolic variant returns a flat int observation; ``image=True`` renders the
    state to pixels.
    """

    def __init__(self, cenv, image=False, image_size=64, num_actions=None,
                 obs_shape=None):
        self.cenv = cenv
        self.image = bool(image)
        self.image_size = int(image_size)
        import jax.numpy as jnp  # noqa: F401  (ensures jax present)
        n = int(num_actions if num_actions is not None
                else getattr(cenv, 'num_actions', getattr(cenv, 'action_space', None) and cenv.action_space.n or 12))
        self.action_space = Box(low=0.0, high=float(n - 1), shape=(1,))
        if self.image:
            self.observation_space = Box(0.0, 1.0, (3, self.image_size, self.image_size))
        else:
            shp = tuple(obs_shape) if obs_shape is not None else self._infer_obs_shape(cenv)
            self.observation_space = Box(-np.inf, np.inf, shp)
        self._n = n
        self._state = None
        self._key = None

    @staticmethod
    def _infer_obs_shape(cenv):
        for attr in ('observation_shape', 'obs_shape'):
            v = getattr(cenv, attr, None)
            if v is not None:
                return tuple(np.atleast_1d(np.asarray(v)).tolist())
        # craftax classic default symbolic flat obs
        return (185,)

    def reset(self, seed=None):
        import jax.random as jrandom
        self._key = jrandom.PRNGKey(0 if seed is None else int(seed))
        out = self.cenv.reset(self._key)
        self._state, obs = out if isinstance(out, tuple) and len(out) == 2 else (out, None)
        return self._obs(obs)

    def step(self, action):
        import jax.random as jrandom
        self._key, subkey = jrandom.split(self._key)
        idx = int(np.clip(round(np.asarray(action).flatten()[0]), 0, self._n - 1))
        out = self.cenv.step(subkey, self._state, idx)
        if len(out) == 5:
            self._state, obs, reward, done, info = out
        else:
            self._state, obs, reward, done = out
            info = {}
        reward_f = float(reward)
        term = bool(done)
        return self._obs(obs), reward_f, term, False, dict(info)

    def _obs(self, obs):
        if self.image:
            try:
                img = self.cenv.render(self._state)  # (H,W,3) uint8 or float
                img = np.asarray(img)
            except Exception:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return _to_chw_float01(_resize_nn(_center_crop_square(img), self.image_size))
        return np.asarray(obs, dtype=np.float32).flatten()


def make_craftax(variant='classic', seed=None, image=False, image_size=64,
                 **adapter_kwargs):
    """Build a goal-conditioned wrapper around a Craftax env.

    ``variant``: 'classic' -> ``CraftaxClassicEnv``, else treated as the full
    ``CraftaxEnv``. Symbolic obs by default; ``image=True`` renders pixels.
    Needs jax (often GPU/CPU-compiled on first call, so the first reset is slow).
    """
    try:
        import jax  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "jax + craftax are required for Craftax envs.\n"
            "  pip install jax jaxlib craftax"
        ) from e
    try:
        if variant == 'classic':
            from craftax.envs.craftax_classic.envs import CraftaxClassicEnv
            cenv = CraftaxClassicEnv()
        else:
            from craftax.envs.craftax_env import CraftaxEnv
            cenv = CraftaxEnv()
    except ImportError as e:
        raise ImportError(
            "craftax is not installed.\n"
            "  pip install craftax\n"
            "(also requires jax / jaxlib)"
        ) from e
    gym_env = _CraftaxToGym(cenv, image=image, image_size=image_size)
    adapter_kwargs.setdefault('success_mode', 'native')
    return GymAdapter(gym_env, image=image, seed=seed, **adapter_kwargs)


__all__ = [
    'GymAdapter', 'make_dmc', 'make_atari', 'make_craftax',
]
