"""worldmodel — a lightweight, dependency-minimal world-model research framework.

Reimplemented from scratch by referencing the design of stable-worldmodel
(galilai-group/stable-worldmodel, MIT). Only depends on torch + numpy.

Pipeline:
    1. collect : roll a policy through an Env to fill a ReplayBuffer
    2. train   : fit a world model (JEPA-style latent prediction)
    3. evaluate: plan with a solver (CEM) + WorldModelPolicy, report success_rate

The world model implements the Costable protocol (``get_cost``) so any solver
can score action candidates against a goal. CTM is wired in as one possible
encoder (``worldmodel.wm.ctm_wm``) so it can be benchmarked against a plain
JEPA encoder (``worldmodel.wm.jepa_wm``) under identical env / data / solver.
"""

from worldmodel import data, envs, solver, wm
from worldmodel.protocols import Actionable, Costable, Transformable
from worldmodel.spaces import Box
from worldmodel.world import World
from worldmodel.policy import PlanConfig, WorldModelPolicy, RandomPolicy

__all__ = [
    'World',
    'PlanConfig',
    'WorldModelPolicy',
    'RandomPolicy',
    'Actionable',
    'Costable',
    'Transformable',
    'Box',
    'data',
    'envs',
    'solver',
    'wm',
]
