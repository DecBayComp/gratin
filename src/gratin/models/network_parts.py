import torch.nn as nn
import torch
import numpy as np
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GlobalAttention, global_mean_pool
from ..data.data_classes import DataModule

## Basic perceptron


class ExpLayer(nn.Module):
    def forward(self, x):
        return torch.exp(x)


def MLP(
    channels, activation="leaky", last_activation="identity", bias=True, dropout=0.0
):

    if last_activation == "identity":
        last_activation = nn.Identity()
    elif last_activation == "exponential":
        last_activation = ExpLayer()

    if activation == "leaky":
        activations = [nn.LeakyReLU(0.2) for i in range(1, len(channels) - 1)] + [
            last_activation
        ]
    else:
        activations = [nn.ReLU() for i in range(1, len(channels) - 1)] + [
            last_activation
        ]
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                nn.BatchNorm1d(channels[i]),
                activations[i - 1],
            )
            for i in range(1, len(channels))
        ]
    )


## base Graph Convolution module


class JumpsConv(MessagePassing):
    def __init__(
        self,
        out_channels,
        x_dim,
        edge_attr_dim,
        dropout=0.0,
        aggr="mean",
        moments=[1],
        f_inner_width=[128, 64],
        g_inner_width=[128, 64],
        **kwargs
    ):
        super(JumpsConv, self).__init__(
            aggr=aggr, **kwargs
        )  # , flow="target_to_source"
        self.out_channels = out_channels
        self.p = dropout
        self.moments = moments
        M = len(moments)
        self.bn_x = nn.BatchNorm1d(x_dim)
        self.net_x = MLP([x_dim, 2 * x_dim, x_dim], last_activation=nn.Tanh())

        self.bn_e = nn.BatchNorm1d(edge_attr_dim)
        self.net_e = MLP(
            [edge_attr_dim, 2 * edge_attr_dim, edge_attr_dim], last_activation=nn.Tanh()
        )

        self.g = MLP(
            [edge_attr_dim + 2 * x_dim] + g_inner_width + [out_channels],
            dropout=dropout,
        )  # ,last_activation = nn.Tanh()))

        if M > 1:
            self.f = nn.Sequential(
                nn.BatchNorm1d(M * out_channels),
                MLP(
                    [M * out_channels] + f_inner_width + [out_channels], dropout=dropout
                ),
            )  # last_activation=nn.Tanh()))
        else:
            self.f = MLP(
                [M * out_channels] + f_inner_width + [out_channels], dropout=dropout
            )

    def forward(self, x, edge_index, edge_attr):

        # print("WARNING : restore bn and net")
        x = self.bn_x(x)
        x = self.net_x(x)

        edge_attr = self.bn_e(edge_attr)
        edge_attr = self.net_e(edge_attr)

        neighbors_message = self.propagate(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        neighbors_message = torch.cat(
            [torch.pow(neighbors_message, m) for m in self.moments], dim=1
        )
        result = self.f(neighbors_message)

        return result

    def message(self, edge_attr, x_j, x_i):
        # X_j is the neighbor node
        # print("X_i receives the message... (showing just time for first layer)")
        # print(x_i[:,-1])
        # print("...coming from x_j (showing just time)")
        # print(x_j[:,-1])
        return self.g(torch.cat([edge_attr, x_j, x_i], dim=1))


class MinimalJumpsConv(MessagePassing):
    def __init__(
        self,
        out_channels,
        x_dim,
        edge_attr_dim,
        dropout=0.0,
        aggr="mean",
        moments=[1],
        f_inner_width=[128, 64],
        **kwargs
    ):
        super(MinimalJumpsConv, self).__init__(
            aggr=aggr, **kwargs
        )  # , flow="target_to_source"
        self.out_channels = out_channels
        self.p = dropout
        self.moments = moments
        M = len(moments) + 1
        self.bn_x = nn.BatchNorm1d(x_dim)
        # self.net_x = MLP([x_dim,2*x_dim,x_dim],last_activation = nn.Tanh())

        MLP_size = [M * x_dim] + f_inner_width + [out_channels]
        if M > 1:
            self.f = nn.Sequential(
                nn.BatchNorm1d(M * x_dim), MLP(MLP_size, dropout=dropout)
            )  # last_activation=nn.Tanh()))
        else:
            self.f = MLP(MLP_size, dropout=dropout)

        nparams = 0
        for n, p in self.named_parameters():
            np_ = np.product(np.array([s for s in p.shape]))
            nparams += np_
        print("f size = ", MLP_size)
        print(
            "Convolution has %d parameters. Input dim is %d, output is %d"
            % (nparams, x_dim, out_channels)
        )

    def forward(self, x, edge_index):  # removed edge_attr

        # print("WARNING : restore bn and net")
        x = self.bn_x(x)
        # x = self.net_x(x)

        neighbors_message = self.propagate(x=x, edge_index=edge_index)

        neighbors_message = torch.cat(
            [x] + [torch.pow(neighbors_message, m) for m in self.moments], dim=1
        )
        result = self.f(neighbors_message)

        return result

    def message(self, x_j):
        # print("message")
        # Should never be called
        return x_j

    def message_and_aggregate(self, adj_t, x):
        # print("message and aggregate")
        return matmul(adj_t, x, reduce=self.aggr)


## Encoder module


class TrajsEncoder(nn.Module):
    """
    Succession of graph convolutions
    Followed by a pooling layer
    And a final MLP that projects to the latent space
    """

    def __init__(
        self,
        dm: DataModule,
        n_c: int = 64,  # Number of convolution kernels
        latent_dim: int = 8,
    ):  # Dimension of edges
        super(TrajsEncoder, self).__init__()
        # To compute moments of features

        x_dim = dm.x_dim
        e_dim = dm.e_dim
        self.no_edge_mode = e_dim == 0

        if self.no_edge_mode:
            Conv = MinimalJumpsConv
        else:
            Conv = JumpsConv

        f_inner_width = [128, 64]
        moments = [1]
        n_final_convolutions = 1

        self.conv1 = Conv(
            out_channels=n_c,
            x_dim=x_dim,
            edge_attr_dim=e_dim,
            f_inner_width=f_inner_width,
            aggr="mean",
            moments=moments,
        )

        self.conv2 = Conv(
            out_channels=n_c,
            x_dim=n_c,
            f_inner_width=f_inner_width,
            edge_attr_dim=e_dim,
            aggr="max",
        )
        final_convs = []
        for i in range(n_final_convolutions):
            final_convs.append(
                Conv(
                    out_channels=n_c,
                    x_dim=(1 + 1 * (i == 0)) * n_c,
                    f_inner_width=f_inner_width,
                    edge_attr_dim=e_dim,
                    aggr="mean",
                )
            )
        self.final_convs = nn.ModuleList(final_convs)

        K = 2 + n_final_convolutions
        # K = 1
        # if params_scarcity == 0:
        gate_nn = MLP([K * n_c, n_c, n_c // 2, 1])
        self.pooling = GlobalAttention(gate_nn=gate_nn)

        self.mlp = MLP([K * n_c, latent_dim])  # used to be tanh for last_activation
        # self.float()
        # print("Final projector has size ", mlp_size)

    def call_conv(self, conv, x, edge_index, edge_attr=None):
        if self.no_edge_mode:
            return conv(x=x, edge_index=edge_index)
        else:
            return conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, data):

        # data.x = data.x.cuda()
        # data.batch = data.batch.cuda()
        if self.no_edge_mode:
            edge_index = data.adj_t  # .cuda()
            data.edge_attr = None
        else:
            edge_index = data.edge_index  # .cuda()
            data.edge_attr = data.edge_attr  # .cuda()

        x_1 = self.call_conv(
            self.conv1, x=data.x, edge_index=edge_index, edge_attr=data.edge_attr
        )

        x_2 = self.call_conv(
            self.conv2, x=x_1, edge_index=edge_index, edge_attr=data.edge_attr
        )

        # Concat two first depths of convolutions
        x = torch.cat([x_1, x_2], dim=1)
        # print(x.shape)
        convolved = [x]
        for i, conv in enumerate(self.final_convs):
            convolved.append(
                self.call_conv(
                    conv,
                    x=convolved[-1],
                    edge_index=edge_index,
                    edge_attr=data.edge_attr,
                )
            )
        # Concat all depths of convolutions
        x = torch.cat(convolved, dim=1)
        # print(x.shape)
        # "average" with attention over all nodes
        x = self.pooling(x=x, batch=data.batch)
        # print(x.shape)
        # Project to latent space
        x = self.mlp(x)

        return x


class AlphaPredictor(nn.Module):
    """
    The only interest of this class is to add an offset and to restrict the output to a plausible interval
    """

    def __init__(
        self,
        p=0.0,
        input_dim=128,
        alpha_fit=False,
        subdiffusive_only=False,
        mlp_size=[128, 128, 64, 16],
    ):
        """
        alpha_fit : whether the latent space has its last dimension indicating TAMSD fit
        """
        super(AlphaPredictor, self).__init__()
        if alpha_fit:
            self.bn_alpha_fit = nn.BatchNorm1d(1)
        self.alpha_fit = alpha_fit
        self.subdiffusive_only = subdiffusive_only
        MLP_size = [input_dim] + mlp_size + [1]
        self.mlp = nn.Sequential(MLP(MLP_size, dropout=p))

        nparams = 0
        for n, p in self.named_parameters():
            np_ = np.product(np.array([s for s in p.shape]))
            # print(n, np_, p.shape)
            nparams += np_
        print("alpha MLP size = ", MLP_size)
        print("Alpha predictor has %d parameters" % nparams)

    def forward(self, x):
        # We only normalize the last column (alpha_fit)
        # as all others have already been batch-normalized at this stage
        if self.alpha_fit:
            x[:, -1:] = self.bn_alpha_fit(x[:, -1:])
        residual = self.mlp(x)  # Last column of x is the alpha fit by MSD
        if self.subdiffusive_only:
            return 0.5 + 0.5 * torch.nn.Tanh()(residual)
        else:
            return 1.0 + 0.99 * torch.nn.Tanh()(residual)
