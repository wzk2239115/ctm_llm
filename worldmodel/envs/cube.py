"""Goal-conditioned CubePush (planar), reimplemented in pure numpy.

A simplified, zero-dependency take on the OGBench cube push task: a free-moving
point agent pushes a square block to a target pose (position + heading) in the
plane. No MuJoCo / OGBench / ogbench dependency — the contact is a hand-rolled
quasi-static push model (translational + angular momentum transfer with
friction decay), so the dynamics never diverge.

Observation / goal / action follow the worldmodel Env protocol (see bench.py)::

    obs = env.reset(seed)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

State (full) = [ax, ay, bx, by, btheta] — agent xy, block xy and block heading.
The goal = [gbx, gby, gbtheta] (target block pose). Success = block pose within
``pos_thresh`` (position) and ``angle_thresh`` (heading) of the goal.

The agent is a point that moves under a 2-D velocity command (``action`` in
[-1, 1]). Whenever it is within ``contact_radius`` of the block centroid it
transfers its motion to the block: the component along the contact normal drives
translation, and the tangential component about the block centroid exerts a
torque that rotates the block (contact-point offset -> angular momentum). The
block's translational and angular velocity then decay under multiplicative
friction. Positions are clipped to ``[0, 1]`` so the dynamics stay bounded.

Two modalities (same design as bench.py / pusht.py):
  * **Full obs**  — state = [ax, ay, bx, by, btheta] (5,). Quasi-Markov (block
    velocity is internal and decays fast). A 1-frame predictor is a reasonable
    baseline.
  * **Partial**   — the block heading btheta is HIDDEN -> state = (4,). To
    predict the next block pose a predictor must infer the heading from the
    translational trajectory, which needs memory; a recurrent predictor
    (stream-ctm) can, a 1-frame Markov predictor cannot. This is the
    manipulation POMDP where stream-ctm is expected to win.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


class CubePush:
    """CubePush: a point agent pushes a square block to a target pose.

    The agent moves freely under a velocity command. On contact (agent within
    ``contact_radius`` of the block centroid) it transfers translational momentum
    (along the agent's motion) and angular momentum (from the tangential
    component about the block centroid) to the block; the block pose then decays
    under multiplicative friction. Positions clipped to ``[0, 1]`` so the system
    is stable and bounded.

    partial obs hides the block heading btheta (POMDP).
    """

    dt = 0.2
    agent_speed = 0.15
    contact_radius = 0.09
    push_gain = 0.8        # agent motion -> block translational velocity
    rot_gain = 6.0         # tangential agent motion -> block angular velocity
    friction = 0.85        # block velocity decay factor (stability)
    max_steps = 200
    pos_thresh = 0.06      # block-to-goal positional success radius
    angle_thresh = 0.35    # block-to-goal orientation success (radians)

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self._agent = np.zeros(2, dtype=np.float32)       # ax, ay
        self._block = np.zeros(3, dtype=np.float32)       # bx, by, btheta
        self._bvel = np.zeros(3, dtype=np.float32)        # vbx, vby, vbtheta
        self._goal = np.zeros(3, dtype=np.float32)        # gbx, gby, gbtheta
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        if self.partial:
            return Box(low=0.0, high=1.0, shape=(4,))  # ax, ay, bx, by
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
        self._block = np.array([
            float(self.rng.uniform(0.2, 0.8)),
            float(self.rng.uniform(0.2, 0.8)),
            float(self.rng.uniform(-np.pi, np.pi)),
        ], dtype=np.float32)
        self._bvel = np.zeros(3, dtype=np.float32)
        if goal is not None:
            self._goal = np.asarray(goal, dtype=np.float32).reshape(3)
        else:
            self._goal = np.array([
                float(self.rng.uniform(0.1, 0.9)),
                float(self.rng.uniform(0.1, 0.9)),
                float(self.rng.uniform(-np.pi, np.pi)),
            ], dtype=np.float32)
        self._step = 0
        return self._obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_agent = self._agent.copy()
        self._agent = np.clip(
            self._agent + self.agent_speed * a * self.dt, 0.0, 1.0
        ).astype(np.float32)
        agent_move = self._agent - prev_agent
        # contact: agent near block centroid transfers momentum
        rel = self._agent - self._block[:2]
        dist = float(np.linalg.norm(rel))
        if dist < self.contact_radius and dist > 1e-6:
            # translational push along the agent's motion direction
            self._bvel[:2] += self.push_gain * agent_move / self.dt
            # rotational push: tangential component of agent motion about block
            cross = float(rel[0] * agent_move[1] - rel[1] * agent_move[0])
            self._bvel[2] += self.rot_gain * cross / (dist + 1e-6)
        # friction decay (stability)
        self._bvel *= self.friction
        # integrate block pose
        self._block[:2] = self._block[:2] + self._bvel[:2] * self.dt
        self._block[2] = (self._block[2] + self._bvel[2] * self.dt + np.pi) % (2 * np.pi) - np.pi
        self._block[:2] = np.clip(self._block[:2], 0.0, 1.0).astype(np.float32)
        self._step += 1
        pdist = float(np.linalg.norm(self._block[:2] - self._goal[:2]))
        adiff = (self._block[2] - self._goal[2] + np.pi) % (2 * np.pi) - np.pi
        terminated = (pdist < self.pos_thresh) and (abs(adiff) < self.angle_thresh)
        truncated = self._step >= self.max_steps
        info = {
            'pos_distance': pdist,
            'angle_err': float(abs(adiff)),
            'agent': self._agent.copy(),
            'block': self._block.copy(),
            'goal': self._goal.copy(),
        }
        return self._obs(), float(terminated), bool(terminated), truncated, info

    def _state_obs(self):
        if self.partial:
            return np.concatenate([self._agent, self._block[:2]]).astype(np.float32)
        return np.concatenate([self._agent, self._block]).astype(np.float32)

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal.copy()}
