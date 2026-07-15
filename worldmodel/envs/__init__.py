"""Goal-conditioned environments for world-model experiments.

Zero-dependency (pure torch/numpy), reimplemented from stable-worldmodel (MIT)
and classic-control textbook dynamics by reference.

Goal-conditioned Env protocol (no gymnasium)::

    env = make_env('tworoom')
    obs = env.reset(seed=0)          # {'pixels'/'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

Registry (``make_env``):
  point-state / point-image        — 2D point reach (smoke)
  tworoom / tworoom-state          — 2-room navigation, wall+door (DINO-WM-style)
  cartpole [-(full|partial)]       — cart-pole; partial hides velocities
  pendulum [-(full|partial)]       — pendulum; partial hides angular velocity
  reacher  [-(full|partial)]       — 2-link reacher; partial hides joint angles
"""

from .point_reach import PointImageReach, PointStateReach
from .bench import TwoRoomNav, CartPole, Pendulum, Reacher
from .mountaincar import MountainCar
from .acrobot import Acrobot
from .swimmer import Swimmer
from .pusht import PushT
from .fetch_push import FetchPush
from .cube import CubePush
from .render import ImageObs, ImagePartial, draw_scene
from collections import deque
import re as _re


def _make_external_env(name: str, **kwargs):
    """Route an external env id (dmc/ atari/ craftax/) to the right factory.

    Honours ``-delayN`` and ``-partial`` / ``-image-partial`` suffixes on top of
    the external env (DelayObs / ImagePartial stack on any env). The underlying
    dependency is imported lazily inside the factory, so a missing dep raises a
    clean ``ImportError`` with a ``pip install`` hint rather than crashing import.
    """
    from .external import make_dmc, make_atari, make_craftax
    from .render import ImagePartial as _ImagePartial

    low = name.lower()
    rest = name.split('/', 1)[1]

    # peel suffixes that apply as wrappers on top of the external env
    delay = 0
    m = _re.search(r'-delay(\d+)$', low)
    if m:
        delay = int(m.group(1))
    partial_mask = low.endswith(('-partial', '-image-partial'))

    # strip all known suffixes to recover the bare env id
    base = _re.sub(r'-(delay\d+|image-partial|image|partial|full)$', '', rest,
                   flags=_re.I)

    # common kwargs forwarded to every factory / GymAdapter
    common = {}
    for k in ('seed', 'image', 'image_size', 'max_steps',
              'goal_thresh', 'goal_noise', 'success_mode', 'reward_thresh'):
        if k in kwargs:
            common[k] = kwargs[k]

    if low.startswith('dmc/'):
        if ':' in base:
            domain, task = base.split(':', 1)
        else:
            domain, task = base, 'run'
        env = make_dmc(domain, task, **common)
    elif low.startswith('atari/'):
        env = make_atari(base, **common)
    elif low.startswith('craftax/'):
        env = make_craftax(base, **common)
    else:
        raise KeyError(f"Unknown external env '{name}'.")

    if partial_mask:
        env = _ImagePartial(env, mask_frac=kwargs.get('mask_frac', 0.3))
    if delay > 0:
        env = DelayObs(env, delay)
    return env


class DenseReward:
    """Wrap env: dense reward = -distance + success_bonus, so PPO gets a continuous
    gradient signal toward the goal (fixes sparse-reward exploration failure on
    hard tasks like manipulation/locomotion). All backbones get the same dense
    reward, so the comparison stays fair."""

    def __init__(self, env, success_bonus=10.0):
        self.env = env
        self.success_bonus = float(success_bonus)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def goal_space(self):
        return self.env.goal_space

    def reset(self, seed=None, goal=None):
        return self.env.reset(seed=seed, goal=goal)

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        dist = info.get('distance', info.get('angle_err', None))
        if dist is None:
            reward = float(term)
        else:
            reward = -float(dist) + (self.success_bonus if term else 0.0)
        return obs, reward, term, trunc, info


class DelayObs:
    """POMDP wrapper: the agent sees the observation from `delay` steps ago.

    Creates a partial-observability task that needs memory: the agent must infer
    the present state from a delayed observation + action history. Stacks on any
    env (e.g. ``pendulum-delay3``).
    """

    def __init__(self, env, delay=3):
        self.env = env
        self.delay = int(delay)
        self._buf = deque(maxlen=self.delay + 1)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def goal_space(self):
        return self.env.goal_space

    def reset(self, seed=None, goal=None):
        obs = self.env.reset(seed=seed, goal=goal)
        self._buf = deque([obs] * (self.delay + 1), maxlen=self.delay + 1)
        return self._buf[0]

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._buf.append(obs)
        return self._buf[0], r, term, trunc, info


def make_env(name: str, **kwargs):
    """Build an env by name. Suffixes:
      ``-partial`` / ``-full``  toggles observability
      ``-delayN``               wraps in DelayObs (obs delayed N steps)

    External envs (lazy deps, see ``worldmodel/envs/external.py``):
      ``dmc/<domain>:<task>``   — dm_control.suite locomotion (e.g. dmc/cheetah:run)
      ``atari/<game>``          — gymnasium ALE (e.g. atari/Pong)
      ``craftax/<variant>``     — craftax (e.g. craftax/classic)
    External ids honour ``-delayN`` and ``-partial`` / ``-image-partial`` suffixes.
    """
    _low = name.lower()
    if _low.startswith(('dmc/', 'atari/', 'craftax/')):
        return _make_external_env(name, **kwargs)

    key = name.lower().replace('_', '-')

    # parse -dense suffix (dense reward shaping for hard exploration tasks)
    dense = key.endswith('-dense')
    if dense:
        key = key[:-len('-dense')]

    # parse -delayN suffix
    import re
    delay = 0
    m = re.search(r'-delay(\d+)$', key)
    if m:
        delay = int(m.group(1))
        key = key[:-len(m.group(0))]

    # parse -image / -image-partial suffix (turns a state env into image obs).
    # NB: skip the canonical 'point-image' env — its name ends in '-image' but
    # it is a native image env, not a state env being promoted; stripping the
    # suffix would turn it into the unknown base 'point'.
    image_mode = None
    if key.endswith('-image-partial') and key != 'point-image':
        image_mode = 'partial'
        key = key[:-len('-image-partial')]
    elif key.endswith('-image') and key != 'point-image':
        image_mode = 'image'
        key = key[:-len('-image')]

    def _split(suffix):
        partial = None
        if key.endswith('-partial'):
            partial = True
        elif key.endswith('-full'):
            partial = False
        base = key[: -(len(suffix))] if partial is not None else key
        return base, partial

    env = None
    if key in ('point-state', 'pointstate', 'state'):
        env = PointStateReach(**kwargs)
    elif key in ('point-image', 'pointimage', 'image'):
        env = PointImageReach(**kwargs)
    elif key.startswith('tworoom'):
        sub = key.split('-')[-1]
        env = TwoRoomNav(image=(sub != 'state'), **kwargs)
    else:
        for base, cls in (('cartpole', CartPole), ('pendulum', Pendulum), ('reacher', Reacher),
                          ('mountaincar', MountainCar), ('acrobot', Acrobot),
                          ('swimmer', Swimmer), ('pusht', PushT),
                          ('fetch-push', FetchPush), ('cube', CubePush)):
            if key.startswith(base):
                partial = None
                if key.endswith('-partial'):
                    partial = True
                elif key.endswith('-full'):
                    partial = False
                env = cls(partial=bool(partial) if partial is not None else False, **kwargs)
                break
    if env is None:
        raise KeyError(f"Unknown env '{name}'.")
    # wrap image obs (state env -> rendered RGB frames)
    if image_mode is not None:
        img_kind = None
        for base in ('cartpole', 'pendulum', 'reacher'):
            if key.startswith(base):
                img_kind = base
                break
        if img_kind is None:
            raise ValueError(f"-image only supported on cartpole/pendulum/reacher, got '{key}'")
        env = ImageObs(env, img_kind, size=kwargs.get('image_size', 32))
        if image_mode == 'partial':
            env = ImagePartial(env, mask_frac=kwargs.get('mask_frac', 0.3))
    if delay > 0:
        env = DelayObs(env, delay)
    return env


__all__ = [
    'PointStateReach', 'PointImageReach', 'TwoRoomNav', 'CartPole', 'Pendulum',
    'Reacher', 'make_env',
]
