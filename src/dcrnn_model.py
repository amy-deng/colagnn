import numpy as np
import torch
import torch.nn as nn

from dcrnn_cell import DCGRUCell
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, args, data): 
        adj_mx = data.orig_adj.cpu().numpy()
        self.max_diffusion_step = 2
        self.cl_decay_steps = 1000
        self.filter_type = 'laplacian'
        self.num_nodes = data.m
        self.num_rnn_layers = 1
        self.rnn_units = args.n_hidden
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, data):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args, data)
        self.input_dim = 1
        self.seq_len = args.window  # for the encoder 
        adj_mx = data.orig_adj.cpu().numpy()
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])
        

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, data):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self) 
        adj_mx = data.orig_adj.cpu().numpy()
        
        Seq2SeqAttrs.__init__(self, args, data)
        self.output_dim = 1
        self.horizon = 1 #args.horizon  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, data):
        super().__init__()
        # if args.cuda:
        #     adj_mx = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        # else:
        #     adj_mx = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        # if args.cuda:
        #     adj_mx = sparse_mx_to_torch_sparse_tensor(data.orig_adj).to_dense().cuda()
        # else:
        #     adj_mx = sparse_mx_to_torch_sparse_tensor(data.orig_adj).to_dense()
        adj_mx = data.orig_adj.cpu().numpy()
        # print(adj_mx)
        Seq2SeqAttrs.__init__(self, args, data)
        self.encoder_model = EncoderModel(args, data)
        self.decoder_model = DecoderModel(args, data)
        self.cl_decay_steps = 1000
        self.use_curriculum_learning = False

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        #  [32, 20, 47]
        # print(inputs.shape)
        inputs = inputs.permute(1,0,2).contiguous()
        encoder_hidden_state = self.encoder(inputs)
        # self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # self._logger.debug("Decoder complete")
        # if batches_seen == 0:
        #     self._logger.info(
        #         "Total trainable parameters {}".format(count_parameters(self))
        #     )
        # print(outputs.shape,'====')
        outputs = outputs.squeeze(0)
        return outputs, None