import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from math import floor
from typing import List, Tuple, Optional, Any
from copy import deepcopy


class DCMF(nn.Module):
    def __init__(self, graph, meta, entity_dimensions, autoencoder_config, reconstructor_config, fusion_config):
        super(DCMF, self).__init__()
        self.graph = graph
        self.meta = meta
        self.entity_dims = entity_dimensions
        self.aec_config = autoencoder_config
        self.rec_config = reconstructor_config
        self.fus_config = fusion_config

        self.autoencoders = nn.ModuleDict()
        self.reconstructors = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        self.__init_model()
        self.dummy = torch.ones(1, requires_grad=True)

    def forward(self, rows, cols, row_ids, col_ids):
        xid = row_ids[0][0]
        row_entities, col_entities, rows, cols = self.emb_forward(
            rows, cols, row_ids, col_ids)
        row_emb, col_emb = self.permute(rows, cols)
        rec = self.rec_forward(xid, row_emb, col_emb)
        return rec, row_entities, col_entities

    def emb_forward(self, rows, cols, row_ids, col_ids):
        xid = row_ids[0][0]
        row_eid, col_eid = self.meta[xid]

        row_entities = self.vae_forward(row_ids, rows)
        col_entities = self.vae_forward(col_ids, cols)
        row_emb = self.fus_forward(row_eid, row_entities)
        col_emb = self.fus_forward(col_eid, col_entities)
        return row_entities, col_entities, row_emb, col_emb

    def vae_forward(self, ids, matrices):
        entities = []
        for (xid, dim), x in zip(ids, matrices):
            entity = {}
            entity['M_bar'], entity['Theta'], entity['Pi'], entity['mu'], entity['logvar'] = self.autoencoders[xid][dim](
                x, self.dummy)
            entities += [entity]
        return entities

    def fus_forward(self, id, entities):
        if len(entities) == 1:
            return entities[0]['mu']
        to_cat = [entity['mu'] for entity in entities]
        fus_mu = self.fusions[id](torch.cat(to_cat, 1))
        return fus_mu

    def rec_forward(self, id, row_emb, col_emb):
        rec = {}
        rec['M_bar'], rec['Theta'], rec['Pi'] = self.reconstructors[id](
            row_emb, col_emb)
        return rec

    def permute(self, row, col):
        r_nrows, _ = row.shape
        c_nrows, _ = col.shape

        row_emb = row.repeat_interleave(c_nrows, dim=0)
        col_emb = col.repeat(r_nrows, 1)
        return row_emb, col_emb

    def __init_model(self):
        """Initiialises all networks in the model."""
        print("Preparing autoencoders' configurations...")
        aec_in_feats = self.__prepare_autoencoder_config()
        print("Preparing reconstructors' configurations...")
        rec_in_feats = self.__prepare_reconstructors_config()
        print("Preparing fusions' configurations...")
        fus_in_feats = self.__prepare_fusion_config()

        print('Initialising autoencoders...')
        self.__init_autoencoders(aec_in_feats)
        print('Initialising reconstructors...')
        self.__init_reconstructors(rec_in_feats)
        print('Initialising fusions...')
        self.__init_fusions(fus_in_feats)

    def __init_autoencoders(self, in_feat_dict):
        """Initialises a pair of autoencoders for each matrix respectively.
        Each pair of autoencoders correspond to the row and column entity of the respective matrices.
        """
        for id, entities in self.meta.items():
            aec_pair = nn.ModuleDict()
            for entity, in_feat in zip(['row', 'col'], in_feat_dict[id]):
                # aec_pair[entity] = VariationalAutoencoder(
                #     in_features=in_feat,
                #     k=self.aec_config['k'],
                #     k_factor=self.aec_config['k_factor'],
                #     hidden_activation=self.aec_config['activation_function'],
                # )
                aec_pair[entity] = VariationalAutoencoder(
                    customEncoder=VariationalEncoder(
                        dimensions=[
                            (in_feat, self.aec_config['hidden_dim']),
                            (self.aec_config['hidden_dim'],
                             self.aec_config['k'])
                        ],
                        hidden_activation=self.aec_config['activation_function']
                    ),
                    customDecoder=VariationalDecoder(
                        dimensions=[
                            (self.aec_config['k'],
                             self.aec_config['hidden_dim']),
                            (self.aec_config['hidden_dim'], in_feat)
                        ],
                        hidden_activation=self.aec_config['activation_function'],
                    ),
                )
            self.autoencoders[id] = aec_pair

    def __init_fusions(self, in_feat_dict):
        """Initialises a data fusion layer for each entity e respectively."""
        for entity in self.graph.keys():
            if in_feat_dict[entity] is not None:
                self.fusions[entity] = DataFusion(
                    in_features=in_feat_dict[entity],
                    factor_features=self.aec_config['k'],
                    activation=self.fus_config['activation_function']
                )

    def __init_reconstructors(self, in_feat_dict):
        """Initialises a reconstructor for each matrix X respectively."""
        for id, _ in self.meta.items():
            self.reconstructors[id] = NMF(
                in_features=2 * self.aec_config['k'],
                activation=self.rec_config['activation_function']
            )

    def __prepare_autoencoder_config(self):
        """Prepares all autoencoder configurations."""
        in_features = {id: (self.entity_dims[col], self.entity_dims[row])
                       for id, (row, col) in self.meta.items()}
        return in_features

    def __prepare_reconstructors_config(self):
        """Prepares all reconstructor configurations."""
        in_features = {id: 2 * self.aec_config['k'] for id in self.meta.keys()}
        return in_features

    def __prepare_fusion_config(self):
        """Prepares all reconstructor configurations."""
        in_features = {}
        for entity, matrices in self.graph.items():
            in_feat = 0
            for id in matrices:
                for e in self.meta[id]:
                    if e == entity:
                        in_feat += self.aec_config['k']
            if in_feat <= self.aec_config['k']:
                in_feat = None
            in_features[entity] = in_feat
        return in_features


class AutoEncoderBase(nn.Module):
    def __init__(self, dimensions: List[Tuple[int]], hidden_activation: str) -> None:
        super(AutoEncoderBase, self).__init__()
        self.activations_available = {
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'selu': nn.SELU(),
            'sigma': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }
        self.dimensions = dimensions
        self.hidden_actf = hidden_activation

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        pass

    def _init_layers(self) -> nn.Sequential:
        """Constructs the hidden layers of the network sequentially."""
        layers = []
        for i, (in_dim, out_dim) in enumerate(self.dimensions[:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self._get_activation_fn(self.hidden_actf))

        return nn.Sequential(*layers)

    def _get_activation_fn(self, activation: str) -> nn.modules.activation:
        if activation not in self.activations_available:
            raise RuntimeError("activation should be one of {}, not {}"
                               .format(self.activations_available.keys(), activation))

        return self.activations_available[activation]

    @property
    def dimensions(self):
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        try:
            for layer_dim in dimensions:
                in_dim, out_dim = layer_dim
        except:
            raise RuntimeError('Each layer dimension should be a pair of ' +
                               '(in_features, out_features), not {}'.format(layer_dim))

        self.__dimensions = dimensions


class VariationalEncoder(AutoEncoderBase):
    def __init__(self, dimensions: List[int], hidden_activation: str = None) -> None:
        super(VariationalEncoder, self).__init__(dimensions, hidden_activation)
        self.hidden_layers = self._init_layers()
        self.bottleneck_layers = self.__init_bottleneck_layers()

    def forward(self, x: Tensor) -> Tuple[float, float]:
        hidden = self.hidden_layers(x)
        return self.bottleneck_layers['mu'](hidden), self.bottleneck_layers['logvar'](hidden)

    def __init_bottleneck_layers(self) -> nn.ModuleDict:
        bottleneck_layer_dim = self.dimensions[-1]
        return nn.ModuleDict({
            'mu': nn.Linear(*bottleneck_layer_dim),
            'logvar': nn.Linear(*bottleneck_layer_dim),
        })


class VariationalDecoder(AutoEncoderBase):
    def __init__(self, dimensions: List[int], hidden_activation: str = 'relu') -> None:
        super(VariationalDecoder, self).__init__(dimensions, hidden_activation)
        self.hidden_layers = self._init_layers()
        self.output_layers = self.__init_output_layers()

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.hidden_layers(x)
        #print(hidden.size())
        M_bar = torch.clamp(
            torch.exp(self.output_layers['M_bar'](hidden)), 1e-5, 1e6)
        Theta = torch.clamp(F.softplus(
            self.output_layers['Theta'](hidden)), 1e-4, 1e4)
        Pi = torch.sigmoid(self.output_layers['Pi'](hidden))
        return M_bar, Theta, Pi

    def __init_output_layers(self) -> nn.ModuleDict:
        output_layer_dim = self.dimensions[-1]
        return nn.ModuleDict({
            'M_bar': nn.Linear(*output_layer_dim),
            'Theta': nn.Linear(*output_layer_dim),
            'Pi': nn.Linear(*output_layer_dim),
        })


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_features: int = 784,
                 k: int = 128, k_factor: float = 0.5,
                 hidden_activation: str = 'relu',
                 customEncoder: nn.Module = None,
                 customDecoder: nn.Module = None) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.dimensions = self.__prepare_layer_dimensions(
            in_features, k, k_factor)
        self.encoder = customEncoder if customEncoder else self.__init_encoder(
            hidden_activation)
        self.decoder = customDecoder if customDecoder else self.__init_decoder(
            hidden_activation)

    def forward(self, x: Tensor, d: Tensor) -> Tuple[Tensor, float, float]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        M_bar, Theta, Pi = self.decoder(z)
        return M_bar, Theta, Pi, mu, logvar

    def reparameterize(self, mu: float, logvar: float) -> float:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def __init_encoder(self, activation: str) -> nn.Module:
        return VariationalEncoder(self.dimensions, activation)

    def __init_decoder(self, activation: str) -> nn.Module:
        dimensions_copy = deepcopy(self.dimensions)
        dimensions_copy.reverse()
        reversed_dimensions = [(t[1], t[0]) for t in dimensions_copy]
        return VariationalDecoder(reversed_dimensions, activation)

    def __prepare_layer_dimensions(self, in_features: int, k: int, k_factor: float) -> Tuple[List[int], List[str]]:
        """Prepares the layer configuration of the model."""
        dimensions = []
        in_dim = in_features
        while True:
            out_dim = floor(in_dim * k_factor)
            if out_dim < k:
                out_dim = k

            dimensions.append((in_dim, out_dim))

            in_dim = out_dim
            if in_dim <= k:  # minimum number of units
                break

        return dimensions


class NMF(nn.Module):
    def __init__(self, in_features: int, activation: str) -> None:
        super(NMF, self).__init__()
        self.activations_available = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'selu': nn.SELU(),
            'sigma': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        })
        self.actf = self.activations_available[activation]

        self.fc1 = nn.Linear(in_features, 200)
        self.fc2 = nn.Linear(200, 100)
        # self.d1 = nn.Dropout(0.2)
        self.fc31 = nn.Linear(100, 1)
        self.fc32 = nn.Linear(100, 1)
        self.fc33 = nn.Linear(100, 1)

    def forward(self, row_emb: Tensor, col_emb: Tensor) -> Tensor:
        #x = torch.cat([row_emb, col_emb], 1)
        #h1 = self.actf(self.fc1(x))
        #h2 = self.actf(self.fc2(h1))
        #print(row_emb.shape)
        #print(col_emb.shape)
        #h2 = torch.matmul(row_emb, torch.transpose(col_emb, 0, 1))
        h2 = torch.sum(row_emb * col_emb, axis = 1) # h2 is of dim row_emb.shape[0] x 1
        # h2 = self.d1(h2)

        #M_bar = torch.clamp(torch.exp(self.fc31(h2)), 1e-5, 1e6)
        #Theta = torch.clamp(torch.exp(self.fc32(h2)), 1e-5, 1e6)
        #Pi = torch.sigmoid(self.fc33(h2))
        M_bar = torch.clamp(torch.exp(h2), 1e-5, 1e6)
        Theta = torch.clamp(torch.exp(h2), 1e-5, 1e6)
        Pi = torch.sigmoid(h2)
        return M_bar, Theta, Pi


class DataFusion(nn.Module):
    def __init__(self, in_features: int, factor_features: int, activation: str) -> None:
        super(DataFusion, self).__init__()
        self.activations_available = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'selu': nn.SELU(),
            'sigma': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        })
        self.fc1 = nn.Linear(in_features, factor_features)
        self.actf = self.activations_available[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.actf(self.fc1(x))


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.normal_(m.weight, mean=0., std=0.01)
        # torch.nn.init.kaiming_uniform_(
        #     m.weight, mode='fan_in', nonlinearity='relu')  # fan_out bad
        torch.nn.init.kaiming_normal_(
            m.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.xavier_normal_(
        #     m.weight, gain=nn.init.calculate_gain('relu'))
