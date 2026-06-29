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


def make_env(name: str, **kwargs):
    """Build an env by name. Suffix ``-partial`` / ``-full`` toggles observability."""
    key = name.lower().replace('_', '-')

    def _split(suffix):
        partial = None
        if key.endswith('-partial'):
            partial = True
        elif key.endswith('-full'):
            partial = False
        base = key[: -(len(suffix))] if partial is not None else key
        return base, partial

    if key in ('point-state', 'pointstate', 'state'):
        return PointStateReach(**kwargs)
    if key in ('point-image', 'pointimage', 'image'):
        return PointImageReach(**kwargs)

    if key.startswith('tworoom'):
        sub = key.split('-')[-1]
        if sub == 'state':
            return TwoRoomNav(image=False, **kwargs)
        return TwoRoomNav(image=True, **kwargs)

    for base, cls in (('cartpole', CartPole), ('pendulum', Pendulum), ('reacher', Reacher)):
        if key.startswith(base):
            partial = None
            if key.endswith('-partial'):
                partial = True
            elif key.endswith('-full'):
                partial = False
            return cls(partial=bool(partial) if partial is not None else False, **kwargs)

    raise KeyError(f"Unknown env '{name}'.")


__all__ = [
    'PointStateReach', 'PointImageReach', 'TwoRoomNav', 'CartPole', 'Pendulum',
    'Reacher', 'make_env',
]
