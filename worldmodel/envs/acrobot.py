"""Goal-conditioned Acrobot (Sutton/Barto 2-link underactuated pendulum),
reimplemented in pure numpy (no gymnasium dependency).

Follows the worldmodel Env protocol (same style as bench.Pendulum/CartPole)::

    env = Acrobot()
    obs = env.reset(seed=0)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

Two rigid links hang from a fixed pivot; only the joint *between* them is
actuated, so the system is underactuated (like a gymnast on a high bar). The
agent must pump energy by swinging to raise the free end above a target height.
State = (theta1, theta2, theta1_dot, theta2_dot); an angle of 0 means a link
points straight down. Continuous action in [-1,1] is the torque on joint 2.

Dynamics use the Lagrangian equations from Sutton & Barto (the gymnasium
"book" branch), integrated with 4-th order Runge-Kutta (dt=0.2). Only standard
numpy is used; the physics is rewritten, not imported.

Goal = a target end-effector height h = -cos(theta1) - cos(theta1+theta2)
(ranges from -2 hanging down to +2 straight up). success = h >= goal_height
(gymnasium fixes this threshold at 1.0; here it is goal-conditioned). Reward
is sparse: 1.0 on success, 0.0 otherwise (matches bench.py).

* **Full obs** — (theta1, theta2, theta1_dot, theta2_dot). Markov.
* **Partial obs** — both angular velocities are hidden. The next angle depends
  on the unseen velocities, so a Markov predictor is fundamentally limited; a
  recurrent predictor that carries thought across ticks can infer them from
  the angle history. This is where stream-ctm is expected to win.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


class Acrobot:
    """Goal-conditioned acrobot (gym Acrobot 'book' Lagrangian dynamics).

    Constants mirror gymnasium's Acrobot (all SI-ish unitless): both links have
    length/mass 1, centre-of-mass at 0.5, moment of inertia 1, gravity 9.8.
    Angular velocities are bounded at +-4*pi (joint 1) and +-9*pi (joint 2).

    partial obs hides both angular velocities -> recurrence must infer them
    from the angle history to predict the swing-up.
    """

    dt = 0.2

    LINK_LENGTH_1 = 1.0
    LINK_LENGTH_2 = 1.0
    LINK_MASS_1 = 1.0
    LINK_MASS_2 = 1.0
    LINK_COM_POS_1 = 0.5
    LINK_COM_POS_2 = 0.5
    LINK_MOI = 1.0

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    max_steps = 200

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self._s = np.zeros(4, dtype=np.float32)  # theta1, theta2, dtheta1, dtheta2
        self._goal_height = 1.0
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        if self.partial:
            return Box(low=[-np.pi, -np.pi], high=[np.pi, np.pi], shape=(2,))
        return Box(low=[-np.pi, -np.pi, -self.MAX_VEL_1, -self.MAX_VEL_2],
                   high=[np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2], shape=(4,))

    @property
    def goal_space(self):
        return Box(low=-1.0, high=2.0, shape=(1,))

    def _height(self, s):
        return float(-np.cos(s[0]) - np.cos(s[0] + s[1]))

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._s = self.rng.uniform(-0.1, 0.1, 4).astype(np.float32)
        self._goal_height = float(self.rng.uniform(0.8, 1.0))
        if goal is not None:
            self._goal_height = float(np.asarray(goal, dtype=np.float32).reshape(-1)[0])
        self._step = 0
        return self._obs()

    def _dsdt(self, s_aug):
        """Derivative of the augmented state [theta1, theta2, d1, d2, torque].

        Sutton & Barto 'book' Lagrangian (matches gymnasium's default branch).
        """
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_aug[-1]
        theta1, theta2, dtheta1, dtheta2 = s_aug[0], s_aug[1], s_aug[2], s_aug[3]
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
                + phi2)
        ddtheta2 = (a + d2 / d1 * phi1
                    - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2)
                    - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0], dtype=np.float64)

    def _rk4(self, s_aug):
        """One 4-th order Runge-Kutta step of size ``dt`` over the dynamics."""
        dt = self.dt
        k1 = self._dsdt(s_aug)
        k2 = self._dsdt(s_aug + dt / 2.0 * k1)
        k3 = self._dsdt(s_aug + dt / 2.0 * k2)
        k4 = self._dsdt(s_aug + dt * k3)
        ns = s_aug + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return ns[:4]

    @staticmethod
    def _wrap(x, m, M):
        """Wrap scalar x into [m, M] (periodic), unlike clip which truncates."""
        diff = M - m
        while x > M:
            x -= diff
        while x < m:
            x += diff
        return x

    def step(self, action):
        torque = float(np.clip(action[0], -1.0, 1.0))
        s_aug = np.append(self._s.astype(np.float64), torque)
        ns = self._rk4(s_aug)
        ns[0] = self._wrap(ns[0], -np.pi, np.pi)
        ns[1] = self._wrap(ns[1], -np.pi, np.pi)
        ns[2] = min(max(ns[2], -self.MAX_VEL_1), self.MAX_VEL_1)
        ns[3] = min(max(ns[3], -self.MAX_VEL_2), self.MAX_VEL_2)
        self._s = ns.astype(np.float32)
        self._step += 1
        h = self._height(self._s)
        dist = max(0.0, self._goal_height - h)
        terminated = h >= self._goal_height
        truncated = self._step >= self.max_steps
        return (self._obs(), float(terminated), bool(terminated), truncated,
                {'height': h, 'distance': dist, 'goal_height': self._goal_height})

    def _state_obs(self):
        if self.partial:
            return np.array([self._s[0], self._s[1]], dtype=np.float32)
        return self._s.copy()

    def _goal_obs(self):
        return np.array([self._goal_height], dtype=np.float32)

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal_obs()}
