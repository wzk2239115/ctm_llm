"""Zero-dependency image renderer + POMDP wrappers for state envs.

Turns a low-dim state env (CartPole/Pendulum/Reacher) into an image env: renders
the physical state to a small RGB frame, so a CNN policy / world model sees
pixels instead of vectors. ``ImagePartial`` additionally occludes a patch of the
frame every step -> image POMDP (the agent sees only part of the scene and must
infer the rest, like DINO-WM occlusion tasks).
"""
import numpy as np
from worldmodel.spaces import Box


def _blob(grid, pos, sigma):
    d2 = np.sum((grid - pos) ** 2, axis=-1)
    return np.exp(-0.5 * d2 / (sigma ** 2)).astype(np.float32)


def draw_scene(size, elements):
    """elements: list of {kind:'point'|'line', p / p1,p2 in [0,1]^2, color:[r,g,b]}.
    Returns (3, size, size) float32 in [0,1]."""
    img = np.zeros((3, size, size), dtype=np.float32)
    coords = np.arange(size, dtype=np.float32) / size
    gx, gy = np.meshgrid(coords, coords, indexing='xy')
    grid = np.stack([gx, gy], axis=-1)
    for el in elements:
        c = el.get('color', [1, 1, 1])
        if el['kind'] == 'point':
            b = _blob(grid, np.asarray(el['p']), el.get('sigma', 0.04))
            for ch in range(3):
                img[ch] += b * c[ch]
        else:  # line
            p1, p2 = np.asarray(el['p1']), np.asarray(el['p2'])
            for t in np.linspace(0, 1, el.get('n', 24)):
                b = _blob(grid, p1 * (1 - t) + p2 * t, el.get('sigma', 0.03))
                for ch in range(3):
                    img[ch] += b * c[ch] * 0.5
    return np.clip(img, 0, 1)


# ---- per-env scene elements (physical state -> drawable) ----
def _cartpole_elements(env, goal):
    if goal:
        return [{'kind': 'point', 'p': [0.5 + env._goal_x * 0.12, 0.72], 'color': [1, 0, 0]}]
    x, _, th, _ = env._s
    cx = 0.5 + x * 0.12
    px, py = cx + np.sin(th) * 0.22, 0.72 - np.cos(th) * 0.22
    return [{'kind': 'point', 'p': [cx, 0.72], 'color': [0, 1, 0]},
            {'kind': 'line', 'p1': [cx, 0.72], 'p2': [px, py], 'color': [1, 1, 1]}]


def _pendulum_elements(env, goal):
    th = env._goal_th if goal else env._th
    px, py = 0.5 + np.sin(th) * 0.32, 0.5 - np.cos(th) * 0.32
    return [{'kind': 'line', 'p1': [0.5, 0.5], 'p2': [px, py],
             'color': [1, 0, 0] if goal else [0, 1, 0]}]


def _reacher_elements(env, goal):
    if goal:
        return [{'kind': 'point', 'p': list(0.5 + env._goal * 0.25), 'color': [1, 0, 0]}]
    e1 = np.array([env.l1 * np.cos(env._a1), env.l1 * np.sin(env._a1)])
    e2 = e1 + np.array([env.l2 * np.cos(env._a1 + env._a2), env.l2 * np.sin(env._a1 + env._a2)])
    s = 0.25 / (env.l1 + env.l2)
    elb, hand = 0.5 + e1 * s, 0.5 + e2 * s
    return [{'kind': 'point', 'p': [0.5, 0.5], 'color': [0.3, 0.3, 0.3]},
            {'kind': 'line', 'p1': [0.5, 0.5], 'p2': list(elb), 'color': [0, 1, 0]},
            {'kind': 'line', 'p1': list(elb), 'p2': list(hand), 'color': [0, 1, 0]}]


_RENDERERS = {'cartpole': _cartpole_elements, 'pendulum': _pendulum_elements,
              'reacher': _reacher_elements}


class ImageObs:
    """Wrap a state env: obs['pixels']/['goal'] become rendered RGB frames."""

    def __init__(self, env, kind, size=32):
        self.env = env
        self.kind = kind
        self.image_size = int(size)

    @property
    def observation_space(self):
        return Box(0.0, 1.0, (3, self.image_size, self.image_size))

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def goal_space(self):
        return Box(0.0, 1.0, (3, self.image_size, self.image_size))

    def reset(self, seed=None, goal=None):
        self.env.reset(seed=seed, goal=goal)
        return self._wrap()

    def step(self, action):
        _, r, t, tr, i = self.env.step(action)
        return self._wrap(), r, t, tr, i

    def _wrap(self):
        els_s = _RENDERERS[self.kind](self.env, goal=False)
        els_g = _RENDERERS[self.kind](self.env, goal=True)
        return {'pixels': draw_scene(self.image_size, els_s),
                'goal': draw_scene(self.image_size, els_g)}


class ImagePartial:
    """Image POMDP: occlude a random patch of each frame (goal stays visible)."""

    def __init__(self, image_env, mask_frac=0.3, seed=None):
        self.env = image_env
        self.mask_frac = float(mask_frac)
        self.image_size = image_env.image_size
        self.rng = np.random.default_rng(seed)

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
        self.rng = np.random.default_rng(seed)
        return self._mask(self.env.reset(seed=seed, goal=goal))

    def step(self, action):
        o, r, t, tr, i = self.env.step(action)
        return self._mask(o), r, t, tr, i

    def _mask(self, obs):
        s = max(2, int(self.image_size * self.mask_frac))
        out = {}
        for k, v in obs.items():
            im = v.copy()
            if k == 'pixels':  # only occlude the current frame, keep goal visible
                H = im.shape[1]
                x0 = int(self.rng.integers(0, H - s + 1))
                y0 = int(self.rng.integers(0, H - s + 1))
                im[:, x0:x0 + s, y0:y0 + s] = 0.0
            out[k] = im
        return out
