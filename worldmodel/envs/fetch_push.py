"""Goal-conditioned FetchPush (planar), reimplemented in pure numpy.

A simplified, zero-dependency take on the classic FetchPush manipulation task:
a planar 2-link arm (same forward-kinematics geometry as ``bench.Reacher``)
pushes a square block to a target position. No MuJoCo / gymnasium Fetch — the
contact is a hand-rolled quasi-static push model (momentum transfer + friction
decay), so the dynamics never diverge.

Observation / goal / action follow the worldmodel Env protocol (see bench.py)::

    obs = env.reset(seed)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

State (full) = [ex, ey, bx, by] — the end-effector xy and the block xy. The goal
= [gx, gy] (target block position). Success = block within ``pos_thresh`` of goal.

The arm is driven exactly like ``Reacher``: ``action`` is a joint-velocity
command in [-1, 1] applied to the two revolute joints, and the end-effector
follows ``_ee(a1, a2)`` (l1 = l2 = 0.5, reach radius 1.0). Whenever the
end-effector is within ``contact_radius`` of the block centroid it transfers
its motion to the block (translational push along the ee displacement + a
positional correction that keeps the block on the contact surface); the block
velocity then decays under multiplicative friction.

Two modalities (same design as bench.py / pusht.py):
  * **Full obs**  — state = [ex, ey, bx, by] (4,). Quasi-Markov (block velocity
    is internal and decays fast). A 1-frame predictor is a reasonable baseline.
  * **Partial**   — the block position is HIDDEN -> state = [ex, ey] (2,). The
    block must be inferred from contact (where pushing "stops" / how the ee
    reacts), which requires memory; a recurrent predictor (stream-ctm) can, a
    1-frame Markov predictor cannot. This is the manipulation POMDP where
    stream-ctm is expected to win.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


class FetchPush:
    """FetchPush: a planar 2-link arm pushes a block to a target position.

    The arm reuses ``Reacher``'s 2-link forward kinematics (``_ee``). ``action``
    controls joint angular velocity; the resulting end-effector motion pushes the
    block when in contact. Block velocity decays under friction so the system is
    stable and bounded. Positions are clipped to ``[-1, 1]`` (within arm reach).

    partial obs hides the block position (POMDP): the block must be inferred from
    the contact dynamics.
    """

    # arm geometry (identical to bench.Reacher)
    l1 = 0.5
    l2 = 0.5
    # dynamics
    dt = 0.1
    speed = 1.0           # joint angular speed scale
    contact_radius = 0.08  # ee within this distance of block -> contact
    push_gain = 1.0       # ee displacement -> block velocity
    friction = 0.8        # block velocity decay (stability)
    # task
    max_steps = 100
    pos_thresh = 0.06     # block-to-goal success radius
    workspace = 1.0       # positions clipped to [-workspace, workspace]

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self._a1 = 0.0                       # joint 1 angle
        self._a2 = 0.0                       # joint 2 angle
        self._block = np.zeros(2, dtype=np.float32)   # bx, by
        self._bvel = np.zeros(2, dtype=np.float32)    # block velocity
        self._goal = np.zeros(2, dtype=np.float32)    # gx, gy
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        if self.partial:
            return Box(low=-self.workspace, high=self.workspace, shape=(2,))  # ex, ey
        return Box(low=-self.workspace, high=self.workspace, shape=(4,))      # ex, ey, bx, by

    @property
    def goal_space(self):
        return Box(low=-0.7, high=0.7, shape=(2,))  # gx, gy (within reach)

    def _ee(self, a1, a2):
        """2-link forward kinematics -> end-effector xy (same as bench.Reacher)."""
        x = self.l1 * np.cos(a1) + self.l2 * np.cos(a1 + a2)
        y = self.l1 * np.sin(a1) + self.l2 * np.sin(a1 + a2)
        return np.array([x, y], dtype=np.float32)

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._a1 = float(self.rng.uniform(-1.0, 1.0))
        self._a2 = float(self.rng.uniform(-1.0, 1.0))
        # place block inside arm reach, away from the goal
        self._block = self.rng.uniform(-0.6, 0.6, 2).astype(np.float32)
        self._bvel = np.zeros(2, dtype=np.float32)
        if goal is not None:
            self._goal = np.asarray(goal, dtype=np.float32).reshape(2)
        else:
            self._goal = self.rng.uniform(-0.6, 0.6, 2).astype(np.float32)
        self._step = 0
        return self._obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_ee = self._ee(self._a1, self._a2)
        # joint-space update (same as Reacher)
        self._a1 += a[0] * self.speed * self.dt
        self._a2 += a[1] * self.speed * self.dt
        ee = self._ee(self._a1, self._a2)
        ee_move = ee - prev_ee
        # contact: ee near block centroid transfers momentum + positional correction
        rel = self._block - ee
        dist = float(np.linalg.norm(rel))
        if dist < self.contact_radius and dist > 1e-6:
            n = (rel / dist).astype(np.float32)
            # keep the block on the contact surface (no overlap with ee)
            self._block = self._block + n * (self.contact_radius - dist)
            # transfer ee motion to block (push along ee displacement)
            self._bvel += self.push_gain * ee_move / self.dt
        # friction decay (stability)
        self._bvel *= self.friction
        # integrate block position
        self._block = np.clip(
            self._block + self._bvel * self.dt, -self.workspace, self.workspace
        ).astype(np.float32)
        self._step += 1
        dist_goal = float(np.linalg.norm(self._block - self._goal))
        terminated = dist_goal < self.pos_thresh
        truncated = self._step >= self.max_steps
        info = {
            'distance': dist_goal,
            'ee': ee.copy(),
            'block': self._block.copy(),
            'goal': self._goal.copy(),
        }
        return self._obs(), float(terminated), bool(terminated), truncated, info

    def _state_obs(self):
        ee = self._ee(self._a1, self._a2)
        if self.partial:
            return ee.copy()  # ex, ey (block hidden)
        return np.concatenate([ee, self._block]).astype(np.float32)  # ex, ey, bx, by

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal.copy()}
