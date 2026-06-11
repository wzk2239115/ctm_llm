import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_topk_sparsity(x, topk_fraction, stepi):
    """Zero out all but topk_fraction of neurons in x along the last dimension."""
    if topk_fraction >= 1.0:
        return x
    k = max(1, int(x.size(-1) * topk_fraction))
    threshold = torch.topk(x.abs(), k, dim=-1)[0][:, -1:]
    mask = (x.abs() >= threshold).float()
    return x * mask


def get_async_tick_mask(stepi, num_neurons, async_tick_periods, async_tick_phases):
    """Return boolean mask [num_neurons] indicating which neurons are active at stepi.
    Band 0: always active. Band k: active every periods[k] ticks, starting at phase[k].
    """
    mask = torch.zeros(num_neurons, dtype=torch.bool)
    bands = len(async_tick_periods)
    neurons_per_band = num_neurons // bands
    for b in range(bands):
        period = async_tick_periods[b]
        phase = async_tick_phases[b] if async_tick_phases else 0
        if (stepi + phase) % period == 0:
            start = b * neurons_per_band
            end = start + neurons_per_band if b < bands - 1 else num_neurons
            mask[start:end] = True
    return mask


def should_halt(certainties, stepi, min_ticks, halt_threshold, halt_mode):
    """Check if we should stop the forward loop early.
    Returns True when certainty exceeds threshold and we're past min_ticks.
    """
    if stepi < min_ticks:
        return False
    if halt_mode == "none" or halt_threshold <= 0:
        return False
    if halt_mode == "threshold":
        max_cert = certainties.max().item()
        return max_cert >= halt_threshold
    return False


class ReflexHead(nn.Module):
    """Lightweight output head that processes synchronisation_out to produce
    early predictions at specified ticks. Works like an anytime output.
    """
    def __init__(self, synch_size, out_dims, hidden_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(synch_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dims),
        )

    def forward(self, synch):
        return self.proj(synch)


class SuperLinear(nn.Module):
    """Standalone SuperLinear for DifferentiatedMemoryNLM.
    A small 2-layer MLP: Linear -> GELU -> Dropout -> Linear.
    """
    def __init__(self, in_features, out_features, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.net(x)


class DifferentiatedMemoryNLM(nn.Module):
    """Multiple NLM modules with different memory lengths, applied to different
    neuron groups. Fast neurons use short memory, slow neurons use long memory.
    """
    def __init__(self, d_model, memory_lengths, hidden_dims_list, dropout=0.0):
        super().__init__()
        self.num_groups = len(memory_lengths)
        self.memory_lengths = memory_lengths
        neurons_per_group = d_model // self.num_groups
        self.group_sizes = [neurons_per_group] * (self.num_groups - 1) + [d_model - neurons_per_group * (self.num_groups - 1)]

        self.nlms = nn.ModuleList()
        for i in range(self.num_groups):
            mem_len = memory_lengths[i]
            hd = hidden_dims_list[i] if hidden_dims_list else 2
            self.nlms.append(SuperLinear(mem_len * self.group_sizes[i], self.group_sizes[i], hd, dropout))

    def forward(self, state_trace):
        """state_trace: (B, d_model, M) where M may be variable across groups.
        We assume state_trace has the max memory_length, and we index into it.
        """
        results = []
        start = 0
        for i, nlm in enumerate(self.nlms):
            size = self.group_sizes[i]
            mem_len = self.memory_lengths[i]
            trace_slice = state_trace[:, start:start+size, -mem_len:].reshape(state_trace.size(0), -1)
            result = nlm(trace_slice)
            results.append(result)
            start += size
        return torch.cat(results, dim=-1)


def add_all_idea_args(parser):
    """Add all CLI arguments for CTM model ideas to an argparse parser."""
    parser.add_argument("--tick_halt_mode", type=str, default="none", choices=["none", "threshold"],
                        help="Early halt mode for the forward loop")
    parser.add_argument("--tick_halt_threshold", type=float, default=0.0,
                        help="Certainty threshold for halting")
    parser.add_argument("--tick_min_ticks", type=int, default=1,
                        help="Minimum ticks before halting is allowed")

    parser.add_argument("--topk_neurons", type=float, default=1.0,
                        help="Fraction of neurons to keep active (top-k sparsity)")

    parser.add_argument("--async_tick_mode", type=str, default="none", choices=["none", "banded"],
                        help="Async tick mask mode")
    parser.add_argument("--async_tick_periods", type=str, default="1,2,4",
                        help="Comma-separated periods per band")
    parser.add_argument("--async_tick_phases", type=str, default="0,0,0",
                        help="Comma-separated phases per band")

    parser.add_argument("--reflex_head", action="store_true", default=False,
                        help="Enable reflex output head for anytime predictions")
    parser.add_argument("--reflex_weight", type=float, default=0.1,
                        help="Weight for reflex head loss")
    parser.add_argument("--reflex_ticks", type=int, default=1,
                        help="First N ticks to apply reflex supervision")
    parser.add_argument("--reflex_distill", action="store_true", default=True,
                        help="Use KL distillation for reflex head training")

    parser.add_argument("--diff_memory", action="store_true", default=False,
                        help="Enable differentiated memory NLMs")
    parser.add_argument("--diff_memory_lengths", type=str, default="4,8,16",
                        help="Comma-separated memory lengths per group")

    parser.add_argument("--draft_mode", type=str, default="none", choices=["none", "revise"],
                        help="Draft-revise mode for two-pass computation")
    parser.add_argument("--draft_block_size", type=int, default=2,
                        help="Number of ticks for draft pass before revision")
    parser.add_argument("--draft_revise_weight", type=float, default=0.0,
                        help="Weight for draft supervision loss")
    parser.add_argument("--draft_corrupt_prob", type=float, default=0.0,
                        help="Probability of state corruption at draft/revision boundary")

    return parser


def apply_draft_revise_corruption(stepi, draft_block_size, activated_state, corrupt_prob, noise_scale=0.1):
    """At draft block boundary, optionally corrupt the state with Gaussian noise.
    Returns (draft_prediction_saved, modified_activated_state).
    """
    saved = None
    if stepi == draft_block_size - 1:
        saved = True
        if corrupt_prob > 0 and torch.rand(1).item() < corrupt_prob:
            noise = torch.randn_like(activated_state) * noise_scale
            activated_state = activated_state + noise
    return saved, activated_state
