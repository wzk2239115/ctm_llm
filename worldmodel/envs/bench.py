"""Zero-dependency benchmark environments, reimplemented from stable-worldmodel
(MIT) and classic-control textbook dynamics by reference (no gymnasium/MuJoCo).

Each is goal-conditioned and follows the worldmodel Env protocol:
    obs = env.reset(seed)            # {'state': ..., 'goal': ...}
    obs, r, term, trunc, info = env.step(action)

Two modalities matter for the stream-ctm vs jepa-mlp comparison:

* **Full obs**  — the observation contains the whole state. Dynamics are
  Markov, so a 1-frame (Markov) predictor is sufficient; stream-ctm ties.
* **Partial obs** — velocities (or joint angles) are HIDDEN. The next state is
  no longer a function of the single observed frame, so a Markov predictor is
  fundamentally limited; a recurrent predictor that carries thought across
  ticks can infer the hidden variables from history. This is where stream-ctm
  is expected to win.

Envs:
  TwoRoomNav  — 2-room navigation with a wall + door (after DINO-WM's TwoRoom).
                Image + state obs; partial by nature (door inferred from pixels).
  CartPole    — classic cart-pole (Sutton/Barto dynamics). partial hides x_dot/theta_dot.
  Pendulum    — classic pendulum swing-up. partial hides theta_dot.
  Reacher     — 2-link kinematic arm reaching a target. partial hides joint angles.
"""

from __future__ import annotations

import numpy as np

from worldmodel.spaces import Box


def _gaussian_image(positions_colors, size=32, sigma=2.0, channels=3):
    """Render a (C,H,W) float32 image in [0,1] with gaussian dots.

    positions_colors: list of (xy_normalized_in_[0,1], rgb_in_[0,1]).
    """
    coords = np.arange(size, dtype=np.float32) / size
    gx, gy = np.meshgrid(coords, coords, indexing='xy')  # (H,W); row=y
    img = np.zeros((channels, size, size), dtype=np.float32)
    for pos, color in positions_colors:
        if pos is None:
            continue
        d2 = (gx - pos[0]) ** 2 + (gy - pos[1]) ** 2
        blob = np.exp(-0.5 * d2 / (sigma ** 2)).astype(np.float32)
        for c in range(channels):
            img[c] = np.maximum(img[c], blob * color[c])
    return img


# ============================== TwoRoom ==============================

class TwoRoomNav:
    """2-room navigation: agent must pass through a door to reach the target.

    Continuous 2D arena [0,1]^2 split by a vertical wall at x=0.5 with one door
    gap. State obs = agent xy (full) ; image obs = top-down render (partial).
    """

    dt = 0.2
    max_speed = 0.15
    threshold = 0.06
    max_steps = 60
    wall_x = 0.5
    wall_half = 0.025

    def __init__(self, image=True, image_size=32, seed=None, partial=False):
        self.image = image
        self.image_size = image_size
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self._agent = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._door_y = 0.5
        self._door_h = 0.15
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        dim = 2  # agent xy (state mode); door is only observable via the image
        return Box(low=0.0, high=1.0, shape=(dim,)) if not self.image else \
            Box(low=0.0, high=1.0, shape=(3, self.image_size, self.image_size))

    @property
    def goal_space(self):
        return Box(low=0.0, high=1.0, shape=(2,))

    def _other_room_goal(self):
        # place goal in the room opposite the agent.
        for _ in range(32):
            g = self.rng.uniform(0.05, 0.95, 2).astype(np.float32)
            if (g[0] < 0.5) != (self._agent[0] < 0.5):  # opposite room
                return g
        return np.array([0.8, 0.5], dtype=np.float32)

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        left = self.rng.random() < 0.5
        ax = self.rng.uniform(0.05, 0.4) if left else self.rng.uniform(0.6, 0.95)
        ay = self.rng.uniform(0.05, 0.95)
        self._agent = np.array([ax, ay], dtype=np.float32)
        self._goal = np.asarray(goal, dtype=np.float32) if goal is not None else self._other_room_goal()
        self._door_y = float(self.rng.uniform(0.2, 0.8))
        self._door_h = float(self.rng.uniform(0.1, 0.2))
        self._step = 0
        return self._obs()

    def _collide(self, prev, nxt):
        # block crossing the wall plane unless within door y-span
        crossed = (prev[0] - self.wall_x) * (nxt[0] - self.wall_x) < 0
        in_door = abs(nxt[1] - self._door_y) < self._door_h
        if crossed and not in_door:
            nxt = nxt.copy()
            nxt[0] = self.wall_x - self.wall_half * np.sign(nxt[0] - self.wall_x) - 1e-3 * np.sign(nxt[0] - self.wall_x)
        return np.clip(nxt, 0.0, 1.0).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        nxt = self._agent + self.max_speed * self.dt * a * 5.0  # scaled to cross room
        self._agent = self._collide(self._agent, nxt)
        self._step += 1
        dist = float(np.linalg.norm(self._agent - self._goal))
        terminated = dist < self.threshold
        truncated = self._step >= self.max_steps
        return self._obs(), float(terminated), terminated, truncated, {'distance': dist}

    def _state_obs(self):
        return self._agent.copy()  # agent xy only; goal is the separate target xy

    def _render(self, pos):
        wall_x_n = self.wall_x
        door_color = np.array([0.2, 0.6, 0.2])
        items = []
        # door gap marker (green) at the wall, at door_y
        items.append((np.array([wall_x_n, self._door_y]), door_color))
        items.append((np.asarray(pos), np.array([1.0, 0.0, 0.0])))   # agent red
        return _gaussian_image(items, self.image_size, sigma=max(1.5, self.image_size / 20))

    def _obs(self):
        if self.image:
            return {'pixels': self._render(self._agent), 'goal': self._render(self._goal)}
        return {'state': self._state_obs(), 'goal': self._goal.copy()}


# ============================== CartPole ==============================

class CartPole:
    """Goal-conditioned cart-pole (Sutton/Barto dynamics).

    Goal = target cart position; success = cart near target AND pole upright.
    partial obs hides velocities (x_dot, theta_dot) -> recurrence must infer them.
    """

    dt = 0.02
    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = 1.1
    length = 0.5
    polemass_length = 0.05
    force_mag = 10.0
    max_steps = 100
    pos_thresh = 0.12
    angle_thresh = 0.10

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self._s = np.zeros(4, dtype=np.float32)  # x, x_dot, theta, theta_dot
        self._goal_x = 0.0
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        return Box(low=-3.0, high=3.0, shape=(2 if self.partial else 4,))

    @property
    def goal_space(self):
        return Box(low=-2.0, high=2.0, shape=(2 if self.partial else 4,))

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._s = np.array([
            self.rng.uniform(-0.5, 0.5), 0.0,
            self.rng.uniform(-0.05, 0.05), 0.0
        ], dtype=np.float32)
        self._goal_x = float(self.rng.uniform(-1.5, 1.5))
        self._step = 0
        return self._obs()

    def _goal_state(self):
        g = np.array([self._goal_x, 0.0, 0.0, 0.0], dtype=np.float32)
        return g

    def step(self, action):
        u = float(np.clip(action[0], -1.0, 1.0)) * self.force_mag
        x, xd, th, thd = self._s
        sinth, costh = np.sin(th), np.cos(th)
        temp = (u + self.polemass_length * sinth * thd ** 2) / self.total_mass
        thacc = (self.gravity * sinth - costh * temp) / \
                (self.length * (4.0 / 3.0 - self.mass_pole * costh ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thacc * costh / self.total_mass
        x = x + self.dt * xd
        xd = xd + self.dt * xacc
        th = th + self.dt * thd
        thd = thd + self.dt * thacc
        self._s = np.array([x, xd, th, thd], dtype=np.float32)
        self._step += 1
        dist = abs(x - self._goal_x)
        terminated = (dist < self.pos_thresh) and (abs(th) < self.angle_thresh)
        truncated = self._step >= self.max_steps
        return self._obs(), float(terminated), bool(terminated), truncated, {'distance': dist, 'angle': abs(th)}

    def _state_obs(self):
        if self.partial:
            return np.array([self._s[0], self._s[2]], dtype=np.float32)  # x, theta
        return self._s.copy()

    def _goal_obs(self):
        g = self._goal_state()
        return np.array([g[0], g[2]], dtype=np.float32) if self.partial else g

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal_obs()}


# ============================== Pendulum ==============================

class Pendulum:
    """Goal-conditioned pendulum (gymnasium dynamics). Goal = target angle.

    partial obs hides angular velocity -> recurrence must infer it to predict.
    """

    dt = 0.05
    g = 10.0
    m = 1.0
    length = 1.0
    max_speed = 8.0
    max_torque = 2.0
    max_steps = 80
    thresh = 0.25

    def __init__(self, partial=False, seed=None):
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self._th = 0.0
        self._thd = 0.0
        self._goal_th = 0.0
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        return Box(low=-1.1, high=1.1, shape=(2 if self.partial else 3,))  # partial: cos,sin

    @property
    def goal_space(self):
        return self.observation_space

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._th = float(self.rng.uniform(-np.pi, np.pi))
        self._thd = float(self.rng.uniform(-1.0, 1.0))
        self._goal_th = float(self.rng.uniform(-np.pi, np.pi))
        self._step = 0
        return self._obs()

    def step(self, action):
        u = float(np.clip(action[0], -1.0, 1.0)) * self.max_torque
        thd = self._thd + (-3 * self.g / (2 * self.length) * np.sin(self._th) + 3 * u / (self.m * self.length ** 2)) * self.dt
        thd = float(np.clip(thd, -self.max_speed, self.max_speed))
        th = self._th + thd * self.dt
        self._th, self._thd = th, thd
        self._step += 1
        # angle difference wrapped to [-pi, pi]
        d = (self._th - self._goal_th + np.pi) % (2 * np.pi) - np.pi
        terminated = (abs(d) < self.thresh) and (abs(self._thd) < 1.0)
        truncated = self._step >= self.max_steps
        return self._obs(), float(terminated), bool(terminated), truncated, {'angle_err': abs(d)}

    def _enc(self, th, thd=None):
        if self.partial:
            return np.array([np.cos(th), np.sin(th)], dtype=np.float32)
        return np.array([np.cos(th), np.sin(th), thd], dtype=np.float32)

    def _obs(self):
        return {'state': self._enc(self._th, self._thd), 'goal': self._enc(self._goal_th, 0.0)}


# ============================== Reacher ==============================

class Reacher:
    """2-link kinematic reacher. Goal = target end-effector xy.

    Observation is the end-effector xy (joint angles hidden) -> a Markov 1-frame
    predictor cannot predict the next ee without knowing the posture, but a
    recurrent predictor can infer posture from the ee trajectory history. This
    is the partial-observability setting where stream-ctm is expected to win.
    """

    l1 = 0.5
    l2 = 0.5
    dt = 0.1
    speed = 1.0
    max_steps = 50
    thresh = 0.06

    def __init__(self, partial=False, seed=None):
        # `partial` accepted for registry symmetry; observation is always ee xy.
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self._a1 = 0.0
        self._a2 = 0.0
        self._goal = np.zeros(2, dtype=np.float32)
        self._step = 0
        self.reset(seed=seed)

    @property
    def observation_space(self):
        return Box(low=-1.1, high=1.1, shape=(2,))

    @property
    def goal_space(self):
        return Box(low=-1.0, high=1.0, shape=(2,))

    def _ee(self, a1, a2):
        x = self.l1 * np.cos(a1) + self.l2 * np.cos(a1 + a2)
        y = self.l1 * np.sin(a1) + self.l2 * np.sin(a1 + a2)
        return np.array([x, y], dtype=np.float32)

    def reset(self, seed=None, goal=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._a1 = float(self.rng.uniform(-1.0, 1.0))
        self._a2 = float(self.rng.uniform(-1.0, 1.0))
        if goal is not None:
            self._goal = np.asarray(goal, dtype=np.float32)
        else:
            self._goal = self._ee(float(self.rng.uniform(-1, 1)), float(self.rng.uniform(-1, 1)))
        self._step = 0
        return self._obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._a1 += a[0] * self.speed * self.dt
        self._a2 += a[1] * self.speed * self.dt
        self._step += 1
        ee = self._ee(self._a1, self._a2)
        dist = float(np.linalg.norm(ee - self._goal))
        terminated = dist < self.thresh
        truncated = self._step >= self.max_steps
        return self._obs(), float(terminated), terminated, truncated, {'distance': dist}

    def _state_obs(self):
        return self._ee(self._a1, self._a2).copy()  # ee xy (joint angles hidden)

    def _obs(self):
        return {'state': self._state_obs(), 'goal': self._goal.copy()}
