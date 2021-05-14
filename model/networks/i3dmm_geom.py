
import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.posEncode import positional_encoding, Sine, sine_init, first_layer_sine_init

class DeformNet(nn.Module):
    def __init__(
        self,
        latent_size,
        init_dims,
        dropout=None,
        dropout_prob=0.0,
        init_latent_in=(),
        init_norm_layers=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positionalEncoding=False,
        num_encoding_functions=6,
        activation="relu"
    ):
        super(DeformNet, self).__init__()

        def make_sequence():
            return []
        latent_size = latent_size
        self.positionalEncoding = positionalEncoding
        self.num_encoding_functions = num_encoding_functions
        
        if self.positionalEncoding:
            self.xyz_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.xyz_dims = 3
        self.nl = activation
        dims_init = [latent_size + self.xyz_dims] + init_dims + [3]
        self.num_layers_init = len(dims_init)
        self.norm_layers_init = init_norm_layers
        self.latent_in_init = init_latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers_init - 1):
            if layer in init_latent_in:
                in_dim = dims_init[layer] + dims_init[0]
            else:
                in_dim = dims_init[layer]
                if self.xyz_in_all and layer != self.num_layers_init - 2:
                    in_dim -= self.xyz_dims
            if weight_norm and layer in self.norm_layers_init:
                setattr(
                    self,
                    "lin_init" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim,dims_init[layer+1])),
                )
            else:
                setattr(self, "lin_init" + str(layer), nn.Linear(in_dim, dims_init[layer+1]))
            if (
                (not weight_norm)
                and self.norm_layers_init is not None
                and layer in self.norm_layers_init
            ):
                setattr(self, "bn_init" + str(layer), nn.LayerNorm(out_dim))
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        if self.nl == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = Sine()
            for layer in range(0, self.num_layers_init - 1):
                lin = getattr(self, "lin_init" + str(layer))
                if layer == 0:
                    first_layer_sine_init(lin)
                else:
                    sine_init(lin)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input, write_debug=False):
        xyz = input[:, -3:]
        if self.positionalEncoding:
            xyz = positional_encoding(xyz)
            
        latent_length = input.shape[1]
        latent_init = input[:, :-3]
        if input.shape[1] > 3 and self.latent_dropout:
            # latent_vecs = input[:, :-3]
            latent_init = F.dropout(latent_init, p=0.2, training=self.training)
            x = torch.cat([latent_init, xyz], 1)
        else:
            x = torch.cat([latent_init, xyz], 1)
        for layer in range(0, self.num_layers_init - 1):
            lin = getattr(self, "lin_init" + str(layer))
            if layer in self.latent_in_init:
                x = torch.cat([x, latent_init, xyz], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer < self.num_layers_init - 2:
                if (
                    self.norm_layers_init is not None
                    and layer in self.norm_layers_init
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn_init" + str(layer))
                    x = bn(x)
                x = self.activation(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x


class RefNet(nn.Module):
    def __init__(
            self,
            sdf_dims,
            dropout=None,
            dropout_prob=0.0,
            sdf_norm_layers=(),
            weight_norm=False,
            xyz_in_all=None,
            use_tanh=False,
            latent_dropout=False,
            positionalEncoding=False,
            num_encoding_functions = 6,
            activation="relu"
    ):
        super(RefNet, self).__init__()

        def make_sequence():
            return []

        self.positionalEncoding = positionalEncoding
        self.num_encoding_functions = num_encoding_functions
        self.nl = activation
        if self.positionalEncoding:
            self.xyz_dims = 3 + 3 * 2 * self.num_encoding_functions
        else:
            self.xyz_dims = 3
            
        dims_sdf = [512] + sdf_dims + [1]
        self.num_layers_sdf = len(dims_sdf)
        self.norm_layers_sdf = sdf_norm_layers

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers_sdf - 1):
            out_dim = dims_sdf[layer + 1]
            if weight_norm and layer in self.norm_layers_sdf:
                setattr(
                    self,
                    "lin_sdf" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims_sdf[layer], out_dim)),
                )
            else:
                setattr(self, "lin_sdf" + str(layer), nn.Linear(dims_sdf[layer], out_dim))

            if (
                    (not weight_norm)
                    and self.norm_layers_sdf is not None
                    and layer in self.norm_layers_sdf
            ):
                setattr(self, "bn_sdf" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        if self.nl == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = Sine()
            for layer in range(0, self.num_layers_sdf - 1):
                lin = getattr(self, "lin_sdf" + str(layer))
                if layer == 0:
                    first_layer_sine_init(lin)
                else:
                    sine_init(lin)
                    


        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        x = input
        if self.positionalEncoding:
            x = positional_encoding(x)
        const = torch.zeros(x.shape[0], 512 - self.xyz_dims, dtype=torch.float32, requires_grad=False).cuda()
        y = torch.cat([x, const], dim=1)

        for layer in range(0, self.num_layers_sdf - 1):
            lin = getattr(self, "lin_sdf" + str(layer))
            y = lin(y)
            # last layer Tanh
            if layer < self.num_layers_sdf - 2:
                if (
                        self.norm_layers_sdf is not None
                        and layer in self.norm_layers_sdf
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn_sdf" + str(layer))
                    y = bn(y)
                y = self.activation(y)
                if self.dropout is not None and layer in self.dropout:
                    y = F.dropout(y, p=self.dropout_prob, training=self.training)
        if hasattr(self, "th"):
            y = self.th(y)
        return y
