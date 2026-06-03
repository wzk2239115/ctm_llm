class CTMLLMConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 6400)
        self.hidden_size = kwargs.get('hidden_size', 512)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.dropout = kwargs.get('dropout', 0.0)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)

        self.d_model = kwargs.get('d_model', 256)
        self.d_input = kwargs.get('d_input', 128)
        self.iterations = kwargs.get('iterations', 3)
        self.memory_length = kwargs.get('memory_length', 5)
        self.heads = kwargs.get('heads', 4)
        self.n_synch_out = kwargs.get('n_synch_out', 256)
        self.n_synch_action = kwargs.get('n_synch_action', 256)
        self.synapse_depth = kwargs.get('synapse_depth', 2)
        self.deep_nlms = kwargs.get('deep_nlms', True)
        self.memory_hidden_dims = kwargs.get('memory_hidden_dims', 4)
        self.neuron_select_type = kwargs.get('neuron_select_type', 'random-pairing')
        self.n_random_pairing_self = kwargs.get('n_random_pairing_self', 0)

        self.num_hidden_layers = kwargs.get('num_hidden_layers', 8)
        self.tie_word_embeddings = kwargs.get('tie_word_embeddings', True)

        assert self.d_model >= max(self.n_synch_out, self.n_synch_action), \
            f"d_model({self.d_model}) must >= n_synch_out({self.n_synch_out}) and n_synch_action({self.n_synch_action})"
        assert self.d_input % self.heads == 0, \
            f"d_input({self.d_input}) must be divisible by heads({self.heads})"

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"CTMLLMConfig({items})"
