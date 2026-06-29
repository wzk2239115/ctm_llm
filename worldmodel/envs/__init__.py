"""Goal-conditioned environments for world-model experiments.

Two minimal, dependency-free point-navigation envs sharing 2D dynamics:

* :class:`PointStateReach` — vector observation (agent xy). Fast; ideal for
  smoke-testing the whole collect -> train -> CEM -> evaluate pipeline.
* :class:`PointImageReach` — RGB image observation (a Gaussian "dot" rendered
  with torch). Used for the real CTM-vs-JEPA comparison, since CTM's strength
  is perceptual / iterative reasoning over images.

Both expose the same minimal Env interface used throughout ``worldmodel``::

    env = PointStateReach()
    obs = env.reset(seed=0)          # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

The observation is the *current state* (controllable); ``'goal'`` is the
target state. A world model predicts future-state latents and a solver drives
the predicted state towards the goal latent.
"""

from .point_reach import PointImageReach, PointStateReach, make_env

__all__ = ['PointStateReach', 'PointImageReach', 'make_env']
