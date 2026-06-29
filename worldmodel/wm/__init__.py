"""World models + builders.

The :class:`~worldmodel.wm.world_model.WorldModel` is encoder-agnostic; the
builders here assemble the two encoders we compare:

* :func:`build_jepa_wm` — plain CNN (image) / MLP (state) encoder.
* :func:`build_ctm_wm` — a Continuous Thought Machine as the image encoder.
"""

from .world_model import WorldModel
from .encoders import CNNEncoder, CTMEncoder, MLPEncoder
from .predictors import MLPPredictor
from .ctm_wm import build_ctm_wm
from .jepa_wm import build_jepa_wm

__all__ = [
    'WorldModel',
    'CNNEncoder',
    'CTMEncoder',
    'MLPEncoder',
    'MLPPredictor',
    'build_ctm_wm',
    'build_jepa_wm',
]
