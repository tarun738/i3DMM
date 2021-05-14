
import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.posEncode import positional_encoding, Sine, sine_init, first_layer_sine_init


class ColorDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        color_dims,
        dropout=None,
        dropout_prob=0.0,
        init_latent_in=(),
        color_latent_in=(),
        color_norm_layers=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positionalEncoding=False,
        num_encoding_functions=6,
        activation="relu"
    ):
        super(ColorDecoder, self).__init__()

        def make_sequence():
            return []
        self.positionalEncoding = positionalEncoding
        self.num_encoding_functions = num_encoding_functions
        self.nl = activation
        if self.positionalEncoding:
            self.xyz_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.xyz_dims = 3
        latent_size = int(latent_size)
        dims_color = [latent_size + self.xyz_dims] + color_dims
        self.num_layers_color = len(dims_color)
        self.norm_layers_color = color_norm_layers
        self.latent_in_color = color_latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers_color - 1):
            if (layer in color_latent_in):
                in_dim = dims_color[layer] + dims_color[0]
            else:
                in_dim = dims_color[layer]
                if self.xyz_in_all and layer != self.num_layers_color - 2:
                    in_dim -= self.xyz_dims

            if weight_norm and layer in self.norm_layers_color:
                setattr(
                    self,
                    "lin_color" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim,dims_color[layer+1])),
                )
            else:
                setattr(self, "lin_color" + str(layer), nn.Linear(in_dim, dims_color[layer+1]))
            if (
                (not weight_norm)
                and self.norm_layers_init is not None
                and layer in self.norm_layers_color
            ):
                setattr(self, "bn_color" + str(layer), nn.LayerNorm(out_dim))

        
        setattr(
            self,
            "lin_color" + "c",
            nn.utils.weight_norm(nn.Linear(dims_color[-1], 3)),
        )
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        if self.nl == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = Sine()
            for layer in range(0, self.num_layers_color - 1):
                lin = getattr(self, "lin_color" + str(layer))
                if layer == 0:
                    first_layer_sine_init(lin)
                else:
                    sine_init(lin)
            lin = getattr(self, "lin_color" + "c")
            sine_init(lin)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]
        if self.positionalEncoding:
            xyz = positional_encoding(xyz)
        latent_length = input.shape[1]
        latent_color = input[:,:-3]
        z = torch.cat([latent_color,xyz],1)
            
        for layer in range(0, self.num_layers_color - 1):
            lin = getattr(self, "lin_color" + str(layer))
            if layer in self.latent_in_color:
                z = torch.cat([z, latent_color, xyz], 1)
            elif layer != 0 and self.xyz_in_all:
                z = torch.cat([z, xyz], 1)
            z = lin(z)
            # last layer Tanh
            if layer < self.num_layers_color - 2:
                if (
                    self.norm_layers_color is not None
                    and layer in self.norm_layers_color
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn_color" + str(layer))
                    z = bn(z)
                z = self.activation(z)
                if self.dropout is not None and layer in self.dropout:
                    z = F.dropout(z, p=self.dropout_prob, training=self.training)
        z = self.activation(z)
        layer_c = getattr(self,"lin_color" + "c")
        col_out = layer_c(z)

        return col_out
