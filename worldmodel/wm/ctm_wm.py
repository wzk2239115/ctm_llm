"""CTM world-model builder — a Continuous ThoughtMachine as the encoder.

This is the CTM-as-world-model baseline. The latent is the CTM's final-tick
synchronisation representation; the latent-dynamics predictor and the rest of
the pipeline are identical to the plain JEPA baseline, so the only varying
factor is the perception/representation front-end.
"""

from __future__ import annotations

from baseline.models.ctm import ContinuousThoughtMachine
from worldmodel.wm.encoders import CTMEncoder
from worldmodel.wm.predictors import MLPPredictor
from worldmodel.wm.world_model import WorldModel


def build_ctm_wm(
    action_dim: int,
    image_size: int = 32,
    iterations: int = 10,
    d_model: int = 256,
    d_input: int = 128,
    heads: int = 4,
    n_synch_out: int = 64,
    n_synch_action: int = 32,
    synapse_depth: int = 2,
    memory_length: int = 10,
    deep_nlms: bool = True,
    memory_hidden_dims: int = 4,
    backbone_type: str = 'resnet18-1',
    predictor_hidden: int = 256,
    cost_mode: str = 'last',
    ema_decay: float = 0.0,
    var_weight: float = 0.0,
) -> WorldModel:
    ctm = ContinuousThoughtMachine(
        iterations=iterations,
        d_model=d_model,
        d_input=d_input,
        heads=heads,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        synapse_depth=synapse_depth,
        memory_length=memory_length,
        deep_nlms=deep_nlms,
        memory_hidden_dims=memory_hidden_dims,
        do_layernorm_nlm=False,
        backbone_type=backbone_type,
        positional_embedding_type='none',
        out_dims=1,  # unused as encoder; minimal output head
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type='random-pairing',
    )
    encoder = CTMEncoder(ctm)
    latent_dim = encoder.latent_dim
    predictor = MLPPredictor(
        latent_dim=latent_dim, action_dim=action_dim, hidden=predictor_hidden
    )
    return WorldModel(
        encoder=encoder,
        predictor=predictor,
        obs_key='pixels',
        action_dim=action_dim,
        cost_mode=cost_mode,
        ema_decay=ema_decay,
        var_weight=var_weight,
    )
