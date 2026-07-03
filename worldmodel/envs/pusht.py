"""Push-T environment (goal-conditioned), reimplemented by reference.

Mirrors the LeWorldModel / diffusion-policy ``PushT`` benchmark: a circular
agent pushes a T-shaped block to a target pose. Hand-rolled quasi-static contact
dynamics instead of a full rigid-body simulator. Zero-dependency (pure numpy).

Observation / goal / action follow the worldmodel Env protocol (see bench.py)::

    obs = env.reset(seed)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

The agent is a point that moves freely under a 2-D velocity command. Whenever it
is within ``contact_radius`` of the block centroid it imparts translational
momentum (along the agent's motion) and angular momentum (from the tangential
component about the centroid); the block then decays under friction. The task is
to bring the block pose (tx, ty, theta) close to a target pose.

Two modalities (same design as bench.py):
  * **Full obs**  — state = [ax, ay, tx, ty, theta] (5,). Markov-predictable.
  * **Partial**   — the block orientation theta is HIDDEN -> state = (4,).
    Predicting the next block pose requires inferring theta from the
    translational trajectory, which needs memory; a recurrent predictor
    (stream-ctm) can, a 1-frame Markov predictor cannot.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


class PushT:
    """Push-T: a circular agent pushes a T-shaped block to a target pose.

    Simplified quasi-static contact (no MuJoCo): the agent moves under a
    velocity command; on contact (agent within ``contact_radius`` of the block
    centroid) it transfers translational + angular momentum to the block, which
    then decays under multiplicative friction. Positions clipped to ``[0, 1]`` so
    the dynamics never diverge.

    partial obs hides the block orientation theta (POMDP).
    """

    dt = 0.2
    agent_speed = 0.15
    contact_radius = 0.09
    push_gain = 0.8       # agent motion -> block translational velocity
    rot_gain = 6.0        # tangential agent motion -> block angular velocity
    friction = 0.8        # block velocity decay factor (stability)
    max_steps = 200
    pos_thresh = 0.06     # block-to-goal positional success radius
    angle_thresh = 0.35   # block-to-goal orientation success (radians)

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self._agent = np.zeros(2, dtype=np.float32)      # ax, ay
        self._t = np.zeros(3, dtype=np.float32)          # tx, ty, theta
        self._tvel = np.zeros(3, dtype=np.float32)       # vtx, vty, vth
        self._goal = np.zeros(3, dtype=np.float32)       # gtx, gty, gtheta
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        if self.partial:
            return Box(low=0.0, high=1.0, shape=(4,))  # ax, ay, tx, ty
        low = np.array([0.0, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, np.pi], dtype=np.float32)
        return Box(low=low, high=high)

    @property
    def goal_space(self):
        low = np.array([0.0, 0.0, -np.pi], dtype=np.float32)
        high = np.array([1.0, 1.0, np.pi], dtype=np.float32)
        return Box(low=low, high=high)

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._agent = self.rng.uniform(0.1, 0.9, 2).astype(np.float32)
        self._t = np.array([
            self.rng.uniform(0.2, 0.8),
            self.rng.uniform(0.2, 0.8),
            self.rng.uniform(-np.pi, np.pi),
        ], dtype=np.float32)
        self._tvel = np.zeros(3, dtype=np.float32)
        if goal is not None:
            self._goal = np.asarray(goal, dtype=np.float32)
        else:
            self._goal = np.array([
                self.rng.uniform(0.1, 0.9),
                self.rng.uniform(0.1, 0.9),
                self.rng.uniform(-np.pi, np.pi),
            ], dtype=np.float32)
        self._step = 0
        return self._obs()

    def _state_obs(self):
        if self.partial:
            return np.concatenate([self._agent, self._t[:2]]).astype(np.float32)
        return np.concatenate([self._agent, self._t]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_agent = self._agent.copy()
        self._agent = np.clip(
            self._agent + self.agent_speed * a * self.dt, 0.0, 1.0
        ).astype(np.float32)
        agent_move = self._agent - prev_agent
        # contact: agent near block centroid transfers momentum
        rel = self._agent - self._t[:2]
        dist = float(np.linalg.norm(rel))
        if dist < self.contact_radius and dist > 1e-6:
            # translational push along the agent's motion direction
            self._tvel[:2] += self.push_gain * agent_move / self.dt
            # rotational push: tangential component of agent motion about block
            cross = float(rel[0] * agent_move[1] - rel[1] * agent_move[0])
            self._tvel[2] += self.rot_gain * cross / (dist + 1e-6)
        # friction decay (stability)
        self._tvel *= self.friction
        # integrate block pose
        self._t[:2] = self._t[:2] + self._tvel[:2] * self.dt
        self._t[2] = (self._t[2] + self._tvel[2] * self.dt + np.pi) % (2 * np.pi) - np.pi
        self._t[:2] = np.clip(self._t[:2], 0.0, 1.0).astype(np.float32)
        self._step += 1
        pdist = float(np.linalg.norm(self._t[:2] - self._goal[:2]))
        adiff = (self._t[2] - self._goal[2] + np.pi) % (2 * np.pi) - np.pi
        terminated = (pdist < self.pos_thresh) and (abs(adiff) < self.angle_thresh)
        truncated = self._step >= self.max_steps
        info = {'pos_distance': pdist, 'angle_err': float(abs(adiff))}
        return self._obs(), float(terminated), bool(terminated), truncated, info

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal.copy()}
