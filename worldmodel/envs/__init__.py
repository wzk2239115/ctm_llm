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
from collections import deque


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
    """
    key = name.lower().replace('_', '-')

    # parse -delayN suffix
    import re
    delay = 0
    m = re.search(r'-delay(\d+)$', key)
    if m:
        delay = int(m.group(1))
        key = key[:-len(m.group(0))]

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
        for base, cls in (('cartpole', CartPole), ('pendulum', Pendulum), ('reacher', Reacher)):
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
    if delay > 0:
        env = DelayObs(env, delay)
    return env


__all__ = [
    'PointStateReach', 'PointImageReach', 'TwoRoomNav', 'CartPole', 'Pendulum',
    'Reacher', 'make_env',
]
