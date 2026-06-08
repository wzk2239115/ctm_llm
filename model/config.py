class CTMLLMConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get('model_type', 'ctm')
        self.vocab_size = kwargs.get('vocab_size', 6400)
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.dropout = kwargs.get('dropout', 0.0)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)

        self.d_model = kwargs.get('d_model', 512)
        self.d_input = kwargs.get('d_input', 256)
        self.iterations = kwargs.get('iterations', 30)
        self.memory_length = kwargs.get('memory_length', 10)
        self.heads = kwargs.get('heads', 8)
        self.n_synch_out = kwargs.get('n_synch_out', 512)
        self.n_synch_action = kwargs.get('n_synch_action', 512)
        self.synapse_depth = kwargs.get('synapse_depth', 3)
        self.deep_nlms = kwargs.get('deep_nlms', True)
        self.memory_hidden_dims = kwargs.get('memory_hidden_dims', 4)
        self.neuron_select_type = kwargs.get('neuron_select_type', 'random-pairing')
        self.n_random_pairing_self = kwargs.get('n_random_pairing_self', 0)

        self.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        self.tie_word_embeddings = kwargs.get('tie_word_embeddings', True)

        self.self_cond = kwargs.get('self_cond', True)
        self.cross_layer_state = kwargs.get('cross_layer_state', True)
        self.block_size = kwargs.get('block_size', 4)
        self.tick_loss_mode = kwargs.get('tick_loss_mode', 'min_conf')
        self.elf_horizon_mode = kwargs.get('elf_horizon_mode', 'none')
        self.elf_max_horizon = kwargs.get('elf_max_horizon', 4)
        self.tick_improve_weight = kwargs.get('tick_improve_weight', 0.0)
        self.tick_improve_margin = kwargs.get('tick_improve_margin', 0.0)
        self.tick_halt_mode = kwargs.get('tick_halt_mode', 'none')
        self.tick_halt_threshold = kwargs.get('tick_halt_threshold', 0.65)
        self.tick_halt_temperature = kwargs.get('tick_halt_temperature', 0.25)
        self.tick_compute_weight = kwargs.get('tick_compute_weight', 0.0)
        self.cell_sparsity_mode = kwargs.get('cell_sparsity_mode', 'none')
        self.cell_topk = kwargs.get('cell_topk', self.d_model)
        self.cell_sparsity_rescale = kwargs.get('cell_sparsity_rescale', True)
        self.moe_routing_mode = kwargs.get('moe_routing_mode', 'none')
        self.moe_num_experts = kwargs.get('moe_num_experts', 1)
        self.moe_topk_experts = kwargs.get('moe_topk_experts', 1)
        self.moe_shared_experts = kwargs.get('moe_shared_experts', 0)
        self.moe_expert_size = kwargs.get('moe_expert_size', 0)
        self.moe_load_balance_weight = kwargs.get('moe_load_balance_weight', 0.0)
        self.moe_router_entropy_weight = kwargs.get('moe_router_entropy_weight', 0.0)
        self.moe_router_z_loss_weight = kwargs.get('moe_router_z_loss_weight', 0.0)
        self.moe_capacity_factor = kwargs.get('moe_capacity_factor', 1.0)
        self.moe_drop_tokens = kwargs.get('moe_drop_tokens', False)
        self.moe_dispatch_mode = kwargs.get('moe_dispatch_mode', 'dense_mask')
        self.moe_topk_warmup_steps = kwargs.get('moe_topk_warmup_steps', 0)
        self.moe_aux_loss_free_bias = kwargs.get('moe_aux_loss_free_bias', False)
        self.moe_expert_dropout = kwargs.get('moe_expert_dropout', 0.0)
        self.moe_mtp_mode = kwargs.get('moe_mtp_mode', 'none')
        self.moe_mtp_horizons = kwargs.get('moe_mtp_horizons', '')

        self.ttt_layer = kwargs.get('ttt_layer', False)
        self.ttt_hidden_mult = kwargs.get('ttt_hidden_mult', 2)
        self.ttt_gate_init = kwargs.get('ttt_gate_init', -2.0)

        assert self.d_model >= max(self.n_synch_out, self.n_synch_action), \
            f"d_model({self.d_model}) must >= n_synch_out({self.n_synch_out}) and n_synch_action({self.n_synch_action})"
        assert self.d_input % self.heads == 0, \
            f"d_input({self.d_input}) must be divisible by heads({self.heads})"

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"CTMLLMConfig({items})"
