"""Reinforcement-learning routes for the world-model framework.

Route 1 (ppo): CTM as an end-to-end policy (PPO), persistent state across env
steps — the original CTM-RL design intent.

Route 2 (dreamer): CTM world model + actor-critic trained in imagination.
"""
from .ppo import (
    PPOTrainer,
    CTMPolicyNetwork,
    MLPPolicyNetwork,
    CTMPolicyBackbone,
    build_policy,
)
from .dreamer import DreamerTrainer, DreamerWorldModel, Actor as DreamerActor, Critic as DreamerCritic