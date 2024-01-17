import torch
import torchmasked
from torch.masked import masked_tensor, as_masked_tensor
from torch.nn.utils.rnn import pad_sequence
import time

import numpy as np
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tqdm import tqdm

from typing import *

from .utils import fan_out_normal_seed


class AvgPool(torch.nn.Module):
    """Average pooling module."""
    def __init__(self, dim: Optional[int] = None):
        """Initialize.
        
        Parameters
        ----------
        dim: int
            Dimension over which to pool
            (default is None)
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            return torchmasked.masked_mean(x, mask, dim=self.dim)
        return torch.mean(x, dim=self.dim)
    
class SumPool(torch.nn.Module):
    """Summation pooling module."""
    def __init__(self, dim: Optional[int] = None):
        """Initialize.
        
        Parameters
        ----------
        dim: int
            Dimension over which to pool
            (default is None)
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            return torchmasked.masked_sum(x, mask, dim=self.dim)
        return torch.mean(outputs, dim=self.dim)


class MaxPool(torch.nn.Module):
    """Top-k maximum pooling module."""
    def __init__(self, dim: Optional[int] = None, k: int = 1):
        """Initialize.
        
        Parameters
        ----------
        dim: int
            Dimension over which to pool
            (default is None)
        k: int
            Amount of maxima to keep
            (default is 1)
        """
        super().__init__()
        self.dim = dim
        self.k = k
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Get top k-values
        x, indices = torch.topk(x, self.k, dim=self.dim, largest=True)
        # Return mean
        if mask is not None:
            return torch.masked_mean(x, mask, dim=self.dim)
        return torch.mean(x, dim=self.dim)
    
class ExtractTensor(torch.nn.Module):
    """Extract tensor for RNN modules."""
    def __init__(self):
        """Initialize."""
        super().__init__()
        
    def forward(self, x: tuple):
        return x[0]


class TrainableNodeEncoder(torch.nn.Module):
    """Class for the node encoder."""
    def __init__(
            self,
            emb_size: int,
            nodes_dict: Dict[str, int],
            device: Union[torch.device, str] = "cuda",
            zeros: bool = True,
            dropout: float = 0.0
    ):
        """
        Trainable node encoder for the node embeddings,
        supports initial feature vectors (i.e. literal values, e.g. floats or
        sentence/word embeddings).
        The encoder supports nodes of different types, that each have different
        associated feature vectors. Every different "featured" node type should
        have an associated integer identifier.
        
        Parameters
        ----------
        emb_size: int
            Desired embedding width.
        nodes_dict: dict
            Dictionary mapping node types to index
        device (Union[torch.device, str], optional):
            PyTorch device to calculate embeddings on
            (Defaults is "cuda")
        zeros: bool
            Whether to initialize unknown embeddings with zeros or random fan out normalization
            (default is True)
        dropout: float
            Dropout rate
            (default is 0)
        """
        super(TrainableNodeEncoder, self).__init__()
        # Make parameters
        self.emb_size = emb_size
        self.device = device
        self.nodes_dict = nodes_dict
        self.zeros = zeros
        self.dropout = torch.nn.Dropout(dropout)

        # Get number of node types
        num_node_types = len(list(self.nodes_dict.keys()))

        # Make embedding tensor shape
        empty_shape = torch.empty((num_node_types, self.emb_size))
        std = 1.0
        # std = torch.sqrt(1./torch.tensor(self.emb_size))

        # Make initial embeddings
        node_embs_init = torch.nn.init.normal_(
            empty_shape,
            std=std
        )
        # a = torch.sqrt(3./torch.tensor(self.emb_size))
        # node_embs_init = torch.nn.init.uniform_(
        #     empty_shape,
        #     a=-a,
        #     b=a
        # )
        # Convert to parameters
        self.node_embs = torch.nn.Parameter(node_embs_init, requires_grad=True)

    def forward(
            self,
            node_mapping: torch.Tensor,
            init_embs: Optional[torch.FloatTensor] = None,
            node_idx: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Embedds nodes into feature representation.
        
        Parameters
        ----------
        node_mapping: torch.Tensor (N x 2)
            Mapping between node index in graph and node identifier for embedding.
        init_embs: Optional[torch.Tensor]
            Initial embeddings for some nodes that overwrite vocabulary embeddings
            (Defaults to None)
            
        Returns
        -------
        torch.Tensor:
            Initial node representations
        """
        # Number of nodes from mapping 
        num_nodes = node_mapping.size(0)
        # If no initial embeddings are given
        if init_embs is None:
            # Make shape
            empty_shape = torch.empty((num_nodes, self.emb_size))
            # Make empty node embeddings
            if not self.zeros:
                init_embs = torch.nn.init.normal_(
                    empty_shape, 
                    std=torch.sqrt(1./torch.tensor(self.emb_size))
                )
            else:
                init_embs = torch.zeros_like(empty_shape)
        # Mask all nodes that are not in vocabulary away
        node_mask = node_mapping[~torch.eq(node_mapping[:, 1], -1), :]
        # Get node embs
        node_embs = self.node_embs.to(node_mapping.device)
        # Make parameters of non literal nodes
        init_embs[node_mask[:, 0], :] = torch.nn.functional.embedding(
            node_mask[:, 1], 
            node_embs
        )
        # Necessary nodes to device once masked and dropout
        return self.dropout(init_embs).to(self.device)


class TrainableRelationEncoder(torch.nn.Module):
    """Encoder for relationships."""
    def __init__(
            self,
            emb_size: int,
            relations_dict: dict,
            epsilon: float = 10e-3,
            theta: float = 10e-3,
            zeros: bool = False,
            dropout: float = 0.0,
            device: Union[torch.device, str] = "cuda",
            max_len: int = 5000

    ) -> torch.Tensor:
        """
        Trainable relationship encoder.
        
        Parameters
        ----------
        emb_size: int
            Input/Output dimensionality
        relations_dict: dict
            Dictionary mapping relations to relation indices.
        epsilon: float
            Scaling parameter for first order approximation.
        theta: float
            Rotation angle for position encoding
        zeros: bool
            Flag whether to encode unknown relationships as zeros or with random initialization.
        device (Union[torch.device, str], optional):
            PyTorch device to calculate embeddings on. Defaults to "cuda".
            
        Returns
        -------
        torch.Tensor:
            Encoded relations

        Notes
        -----
        * Encodes forward relationships as (I + epsilon * X) and it's inverse relationship as (I - epsilon * X).
          --> This ensures that up to second order in epsilon the forward and inverse multiple to the unit operation.
          --> The dictionary 'relations_dict' for construction should contain only forward or backward or relations.
        """
        super(TrainableRelationEncoder, self).__init__()
        # Get parameters
        self.emb_size = emb_size
        self.device = device
        self.relations_dict = relations_dict
        self.epsilon = epsilon
        self.theta = theta
        self.zeros = zeros
        self.dropout = torch.nn.Dropout(dropout)
        
        # Build relations embeddings
        empty_shape = torch.empty((emb_size, emb_size))
        std = torch.sqrt(1./torch.tensor(self.emb_size))
        init_relation_embs = torch.stack([torch.nn.init.normal_(empty_shape, std=std) for _ in relations_dict.keys()])
        self.relation_embs = torch.nn.Parameter(init_relation_embs, requires_grad=True)

        # Make positional encoding for edges
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        axes = torch.zeros(max_len, emb_size)
        axes[:, 0::2] = torch.sin(position * div_term)
        axes[:, 1::2] = torch.cos(position * div_term)
        axes.to(device)
        Q = self.exponential_map(axes)
        self.register_buffer('Q', Q)
        # Register rotation matrices
        self.register_buffer('axes', axes)
        # Make eye
        self.I = torch.eye(self.emb_size)
    
    def exponential_map(self, axes):
        # Make skew with angle
        S = self.theta * (axes[..., None] - torch.transpose(axes[..., None], 1, 2))
        # Make rotation matrices
        Q = torch.linalg.matrix_exp(S)
        # Return
        return Q

    def forward(self, relation_mapping, node_mapping, edge_index, node_embs):
        """
        Encode relations.

        Parameters
        ----------
        relation_mapping:
            Mapping between relation index in graph and relations dict during construction.
            Even indices will get mapped to + epsilon and odd to - epsilon.
            
        Returns
        -------
            Weight matrix containing embeddings for the relations in relation_mapping.
        """
        # Get rotation matrices
        Q = self.Q[relation_mapping[:, 0]].to(self.device)
        # # Try this out
        # child_indices = edge_index[:, 0][relation_mapping[:, 0]].to(self.device)
        # parent_indices = edge_index[:, 1][relation_mapping[:, 0]].to(self.device)
        # axes = self.axes[child_indices, ...] + self.axes[parent_indices, ...]
        # Q = self.exponential_map(axes)
        # Get weights for relations
        X = self.relation_embs.to(self.device)
        # I + epsilon * X on forward/even edge
        W_even = self.I.to(self.device) + self.epsilon * X
        # I - epsilon * X on backward/odd edge
        W_odd = self.I.to(self.device) - self.epsilon * X
        # Stack odd and even weights
        W = torch.stack([W_even, W_odd], dim=1).view(-1).view(-1, self.emb_size, self.emb_size)
        # Stack all weights per type
        relation_embs = torch.stack([W[i, ...] for i in relation_mapping[:, 1]])
        # Positionally encode with rotation matrices
        relation_embs = torch.transpose(Q, 1, 2)@relation_embs@Q
        # Return weights
        return self.dropout(relation_embs)

    
class TreeRGCNPath(torch.nn.Module):
    """Path based tree encoder module."""
    def __init__(
        self, 
        node_encoder: TrainableNodeEncoder,
        relation_encoder: TrainableRelationEncoder,
        device: str = 'cuda'
    ):
        """
        Initialize encoder.
        
        Parameters
        ----------
        node_encoder: TrainableNodeEncoder
            Initialized node encoder 
        relation_encoder: TrainableRelationEncoder
            Initialized relation encoder
        device: str
            String for device allocation
            (Default is 'cuda')
        """
        super().__init__()
        # Get components
        self.node_encoder = node_encoder
        self.relation_encoder = relation_encoder
        self.device = device

    def get_weights(self, path: list, relation_embs: torch.Tensor):
        """
        Helper function to get weights from path.
        
        Parameters
        ----------
        path: list
            List of relation indices in path
        relation_embs: torch.Tensor
            Encoded relations tensor
            
        Returns
        -------
        torch.Tensor: 
            Multiplied path weights
        """
        # Get relation embeddings
        tensors = [relation_embs[int(idx), ...] for idx in path]
        # Multidot the tensors or eye matrix
        if len(tensors) == 0:
            return torch.eye(relation_embs.size(-1)).to(self.device)
        elif len(tensors) == 1:
            return tensors[0]
        else:
            return torch.linalg.multi_dot(tensors)
        
    def update_relations(
        self, 
        new_pairs: dict,
        relation_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper function to update from new pairs.
        
        Parameters
        ----------
        new_pairs: dict
            Dictionary containing indices of paired relation indices
        relation_embs: torch.Tensor
            Encoded relationships

        Returns
        -------
        torch.Tensor:
            Updated relation embeddings
        """
        
        # Create new relations
        new_relation_embs = {}
        # Make new matrix for each pair consecutively
        if new_pairs:
            for new, old in new_pairs.items():
                # Indices are doubed due to forward and backward relations
                idx_a = int(new / 2)
                idx_b = int(old[0]/2)
                idx_c = int(old[1]/2)
                # Make new relations
                b = relation_embs[idx_b, ...] if idx_b > -1 else new_relation_embs.get(idx_b, torch.eye(relation_embs.size(-1)))
                c = relation_embs[idx_c, ...] if idx_c > -1 else new_relation_embs.get(idx_c, torch.eye(relation_embs.size(-1)))
                # indexing is done from the back of the array and in even numbers
                new_relation_embs[idx_a] = b @ c
            # Stack new relations
            new_relation_embs = torch.stack(list(new_relation_embs.values()), dim=0)
            # Concatenate relations with new relations
            relation_embs = torch.cat([relation_embs, new_relation_embs], dim=0)
        # Return embeddings 
        return relation_embs
            

    def forward(
            self,
            paths: list,
            new_pairs: dict,
            degrees: list,
            node_mapping: torch.Tensor,
            relation_mapping: torch.Tensor,
            edge_index: torch.tensor
    ) -> torch.tensor:
        """Forward pass for tree RGCN module.

        Parameters
        ----------
        paths: List[Tuple[int]]
             List of tuples containing integer indices of each node path
        new_pairs: Dict
            Dictionary containing indices of paired relation indices
        node_mapping: torch.LongTensor
            Tensor containing mapping from nodes to indices in training dictionary
        relation_mapping: torch.LongTensor
            Tensor containing mapping from relations to indices in training dictionary
        Returns
        -------
        torch.Tensor:
            Embeddings of root nodes (batch_size x hidden_size)
        """
        # Node, literal and relation embeddings
        node_embs = self.node_encoder(
            node_mapping=node_mapping
        )
        # Embed relationships
        relation_embs = self.relation_encoder(
            relation_mapping=relation_mapping,
            node_mapping=node_mapping,
            edge_index=edge_index,
            node_embs=node_embs
        )
        # Update relation embeddings
        relation_embs = self.update_relations(new_pairs, relation_embs)
        # Get all weights
        weights = torch.stack([
            self.get_weights(path, relation_embs)# * torch.prod(torch.tensor(degree, dtype=torch.float))**(-1)
            for path, degree in zip(paths, degrees)], dim=0)
        # Path transform nodes
        node_embs = torch.einsum('nj, nij -> ni', node_embs, weights)
        # shuffle to 0 -> N order
        node_embs = node_embs[node_mapping[:, 0], :]
        # Return node embeddings
        return node_embs

    
class TreeNet(torch.nn.Module):
    def __init__(
            self,
            emb_size :int,
            hidden_size: int,
            nodes_dict: dict,
            relations_dict: dict,
            labels: List,
            num_layers: int = 1,
            num_blocks: int = 1,
            num_heads: int = 1,
            time_scale: float = 10e5,
            epsilon: float = 10e-3,
            theta: float = 10e-3,
            zeros: bool = True,
            alpha: Optional[callable] = None,
            dropout: float = 0.2,
            device: str = 'cuda'
    ):
        """Initialize tree network.
        
        Parameters
        ----------
        emb_size: int
            Embedding dimension size
        hidden_size: int
            Hidden size for upscale feed forward layer
        nodes_dict: dict
            Node dictionary for node encoder
        relations_dict: dict
            Relations dictionary for relation encoder
        labels: List[str]
            List of labels for which to add alpha coefficients
        num_layers: int
            Number of alpha coefficient weighing layers
        num_blocks: int
            Number of blocks for root embedding iteration
        num_heads: int
            Number of heads for attention layers
        time_scale: float
            Maximal time scale for positional encoding
        epsilon: float
            Scaling parameter for relation encoder
        theta: float
            Rotation angle parameter
        zeros: bool
            Flag whether to zero initialize passing nodes
        alpha: callable
            Callable function for alpha coefficient calculation
        dropout: float
            Dropout rate
        device: str
            String indicating device to be used
        """
        super().__init__()
        # Set parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.labels = labels
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.time_scale = time_scale
        
        # Make node encoder
        self.node_encoder = TrainableNodeEncoder(
            emb_size=emb_size,
            nodes_dict=nodes_dict,
            zeros=zeros,
            device=device,
            dropout=dropout
        )
        # Make relation encoder
        self.relation_encoder = TrainableRelationEncoder(
            emb_size=emb_size,
            relations_dict=relations_dict,
            epsilon=epsilon,
            theta=theta,
            device=device,
            dropout=dropout
        )
        # Make tree encoder
        self.tree_encoder = TreeRGCNPath(
            self.node_encoder,
            self.relation_encoder,
            self.device
        )
        # # Make upscale feed forward
        # self.upscale_root = torch.nn.ModuleDict({
        #     label: torch.nn.ModuleList(
        #         [
        #             torch.nn.Linear(self.emb_size, self.hidden_size)
        #             for block in range(self.num_blocks)
        #         ]
        #     )
        #     for label in labels
        # })
        # # Make downscale feed forward
        # self.downscale_root = torch.nn.ModuleDict({
        #     label: torch.nn.ModuleList(
        #         [
        #             torch.nn.ModuleList([
        #                 torch.nn.Linear(self.hidden_size, self.emb_size, )
        #                 for layer in range(self.num_layers)
        #             ]) 
        #             for block in range(self.num_blocks)
        #         ]
        #     )
        #     for label in labels
        # })
        # Make upscale feed forward
        self.upscale = torch.nn.ModuleDict({
            label: torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.emb_size, self.hidden_size)
                    for block in range(self.num_blocks)
                ]
            )
            for label in labels
        })
        # Make downscale feed forward
        self.downscale = torch.nn.ModuleDict({
            label: torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.hidden_size, self.emb_size, )
                    for block in range(self.num_blocks)
                ]
            )
            for label in labels
        })
        self.attentions = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.MultiheadAttention(self.emb_size, self.num_heads)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        self.lstms = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.LSTM(emb_size, emb_size)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        self.attention_weights = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.Linear(self.emb_size, self.emb_size // self.num_heads, bias=True)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        self.literal_weights = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.ModuleList([
                    torch.nn.Linear(self.emb_size, self.emb_size // self.num_heads, bias=False)
                    for _ in range(self.num_heads)
                ])
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        self.projection_weights = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        # Normalizations for attention
        self.normalizations_attention = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.LayerNorm(self.emb_size)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        # Normalization for feedforward
        self.normalizations_ff = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.LayerNorm(self.emb_size)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })
        # Normalizations for alpha
        self.normalizations_alpha = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.LayerNorm(self.emb_size)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })   
        # Normalizations for alpha
        self.normalizations_alpha_ff = torch.nn.ModuleDict({
            label: torch.nn.ModuleList([
                torch.nn.LayerNorm(self.emb_size)
                for block in range(self.num_blocks)
            ])
            for label in labels
        })    
        # Positional encoding
        self.positional_encoding = TimePositionalEncoding(self.emb_size, time_scale=self.time_scale)
        # final normalization
        self.final_normalization = torch.nn.ModuleDict({
            label: torch.nn.LayerNorm(self.emb_size)
            for label in labels
        })     
        # Set alpha function
        self.alpha = alpha
        
    def forward(
        self, 
        node_mappings, 
        literal_mappings,
        relation_mappings,
        paths,
        time_masks, 
        edge_indices,
        time_values
    ):
        # Make iterator
        iterator = zip(node_mappings, literal_mappings, relation_mappings, paths, time_masks, edge_indices, time_values)
        # Outputs list per patient
        outputs_out, alphas_out = [], []
        # Iterate over patients
        for node_mapping, literal_mapping, relation_mapping, paths_dict, time_mask, edge_index, times in iterator:
            # Should fix in data pipeline
            if not isinstance(time_mask, torch.Tensor):
                time_mask = torch.Tensor(time_mask.astype(bool)).bool()
            # Put time mask on device
            time_mask = time_mask.to(self.device)
            # Get time count
            time_count = torch.maximum(
                time_mask.float().sum(dim=0, keepdims=False), 
                torch.Tensor([1.0]).to(self.device)
            )[..., None].expand((-1, self.emb_size))
            # Make output and alphas dictionaries
            output_dict, alpha_dict = {}, {}
            # Get root nodes, paths and labels for patient
            for root_node, path_dict in paths_dict.items():
                # Make subdicts for labels
                alpha_dict[root_node], output_dict[root_node] = {}, {}
                # Embedding of path transformed nodes --> # N x D
                embs = self.tree_encoder(
                    path_dict['paths'],
                    path_dict['pairs'],
                    path_dict['degrees'],
                    node_mapping,
                    relation_mapping,
                    edge_index
                )
                # Time mask the embeddings
                # embs = torch.multiply(
                #     embs[:, None, :].expand((-1, time_mask.size(1), -1)), 
                #     time_mask[..., None].expand((-1, -1, embs.size(1)))
                # )
                node_mask = (~torch.eq(node_mapping[:, 1], -1)).to(self.device)[:, None, None]
                # Positional time encoding
                # embs = self.positional_encoding(embs, times)
                # Iterate over labels for this root node
                for label in path_dict['labels']:
                    # Only do prediction if in labels
                    if label in self.labels:
                        # Get alphas
                        alphas = torch.ones_like(literal_mapping) * time_mask
                        # Alpha embedding
                        alpha_mapping = alphas * literal_mapping
                        # Get root emb with ones as coefficients
                        root_emb = torch.permute(alpha_mapping, [1, 0]) @ embs
                        # Scale 
                        root_emb = root_emb / torch.sqrt(time_count + self.emb_size)
                        # / torch.sqrt(torch.Tensor([alpha_mapping.size(0) + self.emb_size]))
                        # Initial norm
                        # root_emb = self.final_normalization[label](root_emb)
                        # Make alpha dict for each iteration
                        alpha_dict[root_node][label] = {}
                        # Iterator over blocks
                        for block in range(self.num_blocks):
                            # Get attention query embeddings
                            attention_embs = embs
                            # Get root node for attention manipulation
                            attention_root_emb = self.normalizations_alpha_ff[label][block](root_emb)
                            # # Square block mask
                            # attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(attention_root_emb.size(0))
                            # # Attend
                            # x, _ =  self.attentions[label][block](
                            #     attention_root_emb, 
                            #     attention_root_emb, 
                            #     attention_root_emb, 
                            #     attn_mask=attn_mask.to(self.device)
                            # )
                            x, _ =  self.lstms[label][block](attention_root_emb)
                            # Add
                            attention_root_emb = self.normalizations_attention[label][block](attention_root_emb + x)
                            # Select last time-step
                            attention_root_emb = attention_root_emb[-1:, ...]
                            # Upscale downscale
                            x = self.downscale[label][block](
                                torch.nn.functional.relu(
                                    self.upscale[label][block](attention_root_emb)
                                )
                            )
                            # Add
                            attention_root_emb = self.normalizations_ff[label][block](attention_root_emb + x)
                            # attention_root_emb = attention_root_emb + x
                            # Initialize storage for alphas
                            alpha_dict[root_node][label][f'layer_{block}'] = {}
                            # Change root embedding with block
                            projected_attention_root_emb = self.attention_weights[label][block](attention_root_emb) # torch.zeros_like(attention_root_emb) 
                            # Enumerate heads
                            for head in range(self.num_heads):
                                # Project literal weights
                                # projected_attention_embs = attention_embs 
                                projected_attention_embs = self.literal_weights[label][block][head](attention_embs)
                                # projected_attention_embs = attention_embs
                                # Re-calculate alpha
                                alphas = self.alpha(
                                    time_mask, projected_attention_embs, projected_attention_root_emb)
                                # Add alphas
                                alpha_dict[root_node][label][f'layer_{block}'][f'head_{head}'] = alphas
                            # Sum alphas per head
                            alphas = torch.sum(
                                torch.stack(
                                    list(alpha_dict[root_node][label][f'layer_{block}'].values()),
                                    dim=0
                                )
                                , dim=0
                            ) 
                            # * node_mask
                            # Alpha mapping
                            alpha_mapping = alphas * literal_mapping
                            # Aggregate
                            x = torch.permute(alpha_mapping, [1, 0]) @ embs
                            # Scale 
                            x = x / torch.sqrt(time_count + self.emb_size)
                            # Add 
                            root_emb = x + root_emb
                        # Make final root embedding    
                        final_root_emb = root_emb # self.final_normalization[label](root_emb)
                        # Add output to predictions
                        output_dict[root_node][label] = final_root_emb / torch.sqrt(torch.Tensor([1 + self.num_blocks]))
            # Append output
            outputs_out.append(output_dict)
            # Append alpha
            alphas_out.append(alpha_dict)
        # Return output and alpha
        return outputs_out, alphas_out


class ModelCombiner(torch.nn.Module):
    """
    Combiner function for encoder and models for all corresponding predictions.
    """
    def __init__(self,
                 encoder: TreeNet,
                 models: torch.nn.ModuleDict,
                 losses: Dict,
                 aggregator: callable,
                 batch_first: bool = True,
                 ):
        super().__init__()
        # Set modules
        self.encoder = encoder
        self.models = models
        # Set losses
        self.losses = losses
        # Set aggregator
        self.aggregator = aggregator
        # Training parameters
        self.batch_first = batch_first
        # Assert if the necessary components are present
        self._assert()

    def _assert(self):
        # Assert that all labels are present for losses and modules
        difference = set(self.encoder.labels).difference(self.losses.keys())
        assert len(difference) == 0, \
            f"Class(es) {difference} are not in losses dictionary and required for tree encoder."
        difference = set(self.encoder.labels).difference(self.models.keys())
        assert len(difference) == 0, \
            f"Class(es) {difference} are not in modules dictionary and required for tree encoder."
        
    @staticmethod
    def _pad(sequence, batch_first, dim=0, mask=False):
        return pad_sequence(sequence, batch_first)
        # if batch_first:
        #     batch_dim = 0
        # else:
        #     batch_dim = 1
        # maximum = max([element.shape[dim] for element in sequence])
        # n_dim = len(sequence[0].shape)
        # seq = []
        # _mask = []
        # for idx, element in enumerate(sequence):
        #     m = torch.ones_like(element)
        #     padding = [0] * (2 * n_dim)
        #     padding[2 * (n_dim - dim - 1)] = maximum - element.shape[dim]
        #     padded = torch.nn.functional.pad(element, padding)
        #     m = torch.nn.functional.pad(m, padding)
        #     seq.append(padded)
        #     _mask.append(m)
        # stacked = torch.stack(seq, dim=batch_dim)
        # masked = torch.stack(_mask, dim=batch_dim)
        # if mask:
        #     return stacked, masked
        # else:
        #     return stacked 
        
    def forward(self, batch):
        # Get necessary tensors
        node_mapping = [b.node_mapping for b in batch]
        literal_mapping = [b.literal_mapping for b in batch]
        relation_mapping = [b.relation_mapping for b in batch]
        paths = [b.paths for b in batch]
        time_masks = [b.time_mask for b in batch]
        edge_indices = [b.edge_index for b in batch]
        time_values = [torch.LongTensor(b.times) for b in batch]
        # Calculate outputs and corresponding alpha values
        outputs, alphas = self.encoder(
            node_mappings=node_mapping,
            literal_mappings=literal_mapping,
            relation_mappings=relation_mapping,
            paths=paths,
            time_masks=time_masks,
            edge_indices=edge_indices,
            time_values=time_values
        )
        # Batch embeddings
        embeddings = {
            label: self._pad([ # pre-padding
                out[label] for output in outputs for out in output.values() if label in out],
                batch_first=self.batch_first,
                mask=False
            )
            for label in self.models.keys()
        }
        # Perform task dependent model
        outputs = {
            label: model(embeddings[label])
            for label, model in self.models.items()
        }
        # Return outputs and alphas
        return {"embeddings": embeddings, "output": outputs, "alpha": alphas}
    
    def get_targets(self, data):
        # Get targets
        targets = [d.labels for d in data]
        # # Batch targets
        targets = {
            label: torch.stack(
                [torch.tensor(out[label]['labels']) for label_dict in targets for out in label_dict.values() if
                 label in out])
            for label in self.models.keys()
        }
        return targets

    def training_step(self, data, batch_idx):
        # Get targets
        targets = self.get_targets(data)
        # # Get times in batch
        # times = batch.times
        # # Get targets in batch
        # targets = batch.labels
        # # Get time masked labels
        # masks = mask_labels(times, labels)
        # Get predictions
        predictions = self(data)
        # Masking
        # preds_ = preds[mask,:]
        # Initialize losses
        losses = []
        # Iterate over labels
        for label, loss_fn in self.losses.items():
            # Calculate loss for label
            loss_ = loss_fn(predictions['output'][label].cpu(), targets[label].float())
            # Propagate loss
            losses.append(loss_)
        # Reduce losses
        loss = self.aggregator(losses)
        # Return loss and alphas
        return {"loss": loss, "alpha": predictions["alpha"]}

    def validation_step(self, data, batch_idx):
        # Get targets
        targets = [d.labels for d in data]
        # Batch targets
        targets = {
            label: torch.stack(
                [torch.tensor(out[label]['labels']) for label_dict in targets for out in label_dict.values() if
                 label in out])
            for label in self.models.keys()
        }
        # # Get times in batch
        # times = batch.times
        # # Get targets in batch
        # targets = batch.labels
        # # Get time masked labels
        # masks = mask_labels(times, labels)
        # Get predictions
        predictions = self(data)
        # Masking
        # preds_ = preds[mask,:]
        # Initialize losses
        losses = []
        # Iterate over labels
        for label, loss_fn in self.losses.items():
            # Calculate loss for label
            loss_ = loss_fn(predictions['output'][label].cpu(), targets[label].float())
            # Propagate loss
            losses.append(loss_)
        # Reduce losses
        loss = self.aggregator(losses)
        
        if batch_idx == 0:
            print("Validation loss @ batch 0: ", loss.item())

        # if self.print_metrics is not None:
        #     print_metrics = {
        #         label: {
        #         metric: metric_fn(predictions['output'][label], targets[label])
        #         for metric, metric_fn in label_metrics.items()
        #         }
        #         for label, label_metrics in self.print_metrics.items()
        #     }
        #     self.log_dict(print_metrics, on_step = False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        # if self.log_metrics is not None:
        #     log_metrics = {
        #         label: {
        #         metric: metric_fn(predictions['output'][label], targets[label])
        #         for metric, metric_fn in label_metrics.items()
        #         }
        #         for label, label_metrics in self.log_metrics.items()
        #     }
        #     self.log_dict(log_metrics, on_step = False, on_epoch = True, prog_bar = False, batch_size=batch_size)

        return {"loss": loss, "predictions": predictions, "targets": targets}


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, time_scale=10000, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        omega = time_scale ** (-torch.arange(0, d_model, 2) / d_model) 
        self.register_buffer('omega', omega)

    def forward(self, x: Tensor, times: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert x.size(1) == times.size(0)
        pe = torch.zeros(1, x.size(1), self.d_model).to(self.device)
        omega = self.omega[None, :].to(self.device)
        time = times[:, None].to(self.device)
        pe[0, :, 0::2] = torch.sin(omega * time) 
        pe[0, :, 1::2] = torch.cos(omega * time)
        x = x + pe * torch.sqrt(1./torch.tensor(self.d_model))
        return self.dropout(x)
    
    
class MLP(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers, n_classes, bias, device: str = 'cuda'):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleDict({
            f'layer_{i}': torch.nn.Linear(in_features=emb_size, out_features=hidden_size, bias=bias)
            for i in range(n_layers)
        })
        self.output_layer = torch.nn.Linear(in_features=hidden_size, out_features=n_classes, bias=bias)
        self.device = device

    def forward(self, x):
        for layer in self.hidden_layers.values():
            x_ = layer(x)
            x_ = torch.nn.functional.relu(x_)
            x = x + x_
        return self.output_layer(x)
    
class printer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x.shape)
        return x
    
def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    # pad = (kernel_size - 1) * dilation
    pad = 'same'
    return torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

class Padding(torch.nn.Module):
    def __init__(self, min_length: int, max_length: int):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        
    def forward(self, x):
        """
        Minimum time padding for embedding
        
        Parameters
        ----------
        x: torch.Tensor
            Input data (T x B x F)
        
        """
        # get time size
        T = x.size(0)
        # Check if long enough
        if T < self.min_length:
            # Pad to minimum length
            x = torch.nn.functional.pad(x, [0, 0, 0, 0, self.min_length - T, 0])
        # Return padded array
        return x   


class ConvolutionNetwork(torch.nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        # Define parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        # Define scale up bottleneck convolution layer
        self.scale_up = torch.nn.Conv1d(self.emb_size, self.hidden_size, 1, padding="same")
        # Define first dilated convolution
        self.convolutions = torch.nn.ModuleDict(
            {
                "layer_one": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=9, dilation=1),
                "layer_two": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=7, dilation=1),
                "layer_three": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=3, dilation=2),
                "layer_four": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=3, dilation=4),
                "layer_five": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=3, dilation=8),
                "layer_six": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=3, dilation=16),
                "layer_seven": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=3, dilation=32),
                "layer_eight": CausalConv1d(self.hidden_size, self.hidden_size, kernel_size=7, dilation=1)
            }
        )
        # dropout
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        # Scale to hidden size
        x = self.scale_up(x)
        # Do convolutions
        for name, conv in self.convolutions.items():
            # Conv layer
            x_ = conv(x)  
            # Residual addition
            x = x+ x_
            # Dropout
            x = self.dropout(x)
            # Relu
            x= torch.nn.ReLU()(x)
        # Return tensor
        return x

###### Legacy
"""
    def forward(
        self, 
        node_mappings, 
        literal_mappings,
        relation_mappings,
        paths,
        time_masks, 
        edge_indices,
        time_values
    ):
        # Make iterator
        iterator = zip(node_mappings, literal_mappings, relation_mappings, paths, time_masks, edge_indices, time_values)
        # Outputs list per patient
        outputs_out, alphas_out = [], []
        # Iterate over patients
        for node_mapping, literal_mapping, relation_mapping, paths_dict, time_mask, edge_index, times in iterator:
            # Should fix in data pipeline
            if not isinstance(time_mask, torch.Tensor):
                time_mask = torch.Tensor(time_mask.astype(bool)).bool()
            # Put time mask on device
            time_mask = time_mask.to(self.device)
            # Get time count
            time_count = torch.maximum(
                time_mask.float().sum(dim=0, keepdims=True), 
                torch.Tensor([1.0]).to(self.device)
            )[..., None].expand((-1, -1, self.emb_size))
            # Make output and alphas dictionaries
            output_dict, alpha_dict = {}, {}
            # Get root nodes, paths and labels for patient
            for root_node, path_dict in paths_dict.items():
                # Make subdicts for labels
                alpha_dict[root_node], output_dict[root_node] = {}, {}
                # Embedding of path transformed nodes --> # N x D
                embs = self.tree_encoder(
                    path_dict['paths'],
                    path_dict['pairs'],
                    path_dict['degrees'],
                    node_mapping,
                    relation_mapping,
                    edge_index
                )
                # Time mask the embeddings
                embs = torch.multiply(
                    embs[:, None, :].expand((-1, time_mask.size(1), -1)), 
                    time_mask[..., None].expand((-1, -1, embs.size(1)))
                )
                node_mask = (~torch.eq(node_mapping[:, 1], -1)).to(self.device)[:, None, None]
                # Positional time encoding
                # embs = self.positional_encoding(embs, times)
                # Iterate over labels for this root node
                for label in path_dict['labels']:
                    # Only do prediction if in labels
                    if label in self.labels:
                        # Scale embeddings with alphas
                        literal_embs = literal_mapping[:, :, None].to(self.device) * embs
                        # Get alphas
                        alphas = torch.ones_like(literal_embs)
                        # Alpha embedding
                        alpha_embs = alphas * literal_embs
                        # Get root emb with ones as coefficients
                        root_emb = torch.sum(alpha_embs / time_count, dim=0)
                        # Make alpha dict for each iteration
                        alpha_dict[root_node][label] = {}
                        # Iterator over blocks
                        for block in range(self.num_blocks):
                            # Get attention query embeddings
                            # attention_embs = alpha_embs
                            attention_embs = literal_embs
                            # # Get root node for attention manipulation
                            attention_root_emb = root_emb
                            # attention_root_emb = root_emb / torch.sqrt(torch.Tensor([block + 1.0])).to(self.device)
                            # Square block mask
                            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(attention_root_emb.size(0))
                            # Attend
                            x, _ =  self.attentions[label][block](
                                attention_root_emb, 
                                attention_root_emb, 
                                attention_root_emb, 
                                attn_mask=attn_mask.to(self.device)
                            )
                            # Add
                            attention_root_emb = self.normalizations_attention[label][block](attention_root_emb + x)
                            # Select last time-step
                            attention_root_emb = attention_root_emb[-1:, ...]
                            # attention_root_emb = attention_root_emb + x
                            # Upscale downscale
                            x = self.downscale[label][block](
                                torch.nn.functional.relu(
                                    self.upscale[label][block](attention_root_emb)
                                )
                            )
                            # Add
                            attention_root_emb = self.normalizations_ff[label][block](attention_root_emb + x)
                            # attention_root_emb = attention_root_emb + x
                            # Initialize heads and storage for alphas
                            concat_embs = []
                            alpha_dict[root_node][label][f'layer_{block}'] = {}
                            # Change root embedding with block
                            projected_attention_root_emb = self.attention_weights[label][block](attention_root_emb)
                            # Enumerate heads
                            for head in range(self.num_heads):
                                # Project literal weights
                                projected_attention_embs = self.literal_weights[label][0][head](attention_embs)
                                # Re-calculate alpha
                                alphas = self.alpha(
                                    time_mask, projected_attention_embs, projected_attention_root_emb)[..., None]
                                alphas = alphas * node_mask
                                # Calculate weighted sum
                                concat_embs.append(alphas * projected_attention_embs)
                                # Add alphas
                                alpha_dict[root_node][label][f'layer_{block}'][f'head_{head}'] = alphas
                            # Concatenate heads
                            alpha_embs = torch.concat(concat_embs, dim=-1)
                            # Mix heads
                            alpha_embs = self.projection_weights[label][block](alpha_embs)
                            # Aggregate paths
                            x = torch.sum(alpha_embs / time_count, dim=0)                            
                            # Add
                            # root_emb = self.normalizations_alpha_ff[label][block](root_emb +  x)
                            root_emb = root_emb + x
                        # Make final root embedding    
                        final_root_emb = root_emb / torch.sqrt(torch.Tensor([self.num_blocks + 1.0])).to(self.device)
                        # Add output to predictions
                        output_dict[root_node][label] = final_root_emb
            # Append output
            outputs_out.append(output_dict)
            # Append alpha
            alphas_out.append(alpha_dict)
        # Return output and alpha
        return outputs_out, alphas_out
"""

"""
# Make iterator
        iterator = zip(node_mappings, literal_mappings, relation_mappings, paths, time_masks, edge_indices, time_values)
        # Outputs list per patient
        outputs_out, alphas_out = [], []
        # Iterate over patients
        for node_mapping, literal_mapping, relation_mapping, paths_dict, time_mask, edge_index, times in iterator:
            # Should fix in data pipeline
            if not isinstance(time_mask, torch.Tensor):
                time_mask = torch.Tensor(time_mask.astype(bool)).bool()
            # Put time mask on device
            time_mask = time_mask.to(self.device)
            # Get time count
            time_count = torch.maximum(
                time_mask.float().sum(dim=0, keepdims=True), 
                torch.Tensor([1.0]).to(self.device)
            )[..., None].expand((-1, -1, self.emb_size))
            # Make output and alphas dictionaries
            output_dict, alpha_dict = {}, {}
            # Get root nodes, paths and labels for patient
            for root_node, path_dict in paths_dict.items():
                # Make subdicts for labels
                alpha_dict[root_node], output_dict[root_node] = {}, {}
                # Embedding of path transformed nodes --> # N x D
                embs = self.tree_encoder(
                    path_dict['paths'],
                    path_dict['pairs'],
                    path_dict['degrees'],
                    node_mapping,
                    relation_mapping,
                    edge_index
                )
                # Time mask the embeddings
                embs = torch.multiply(
                    embs[:, None, :].expand((-1, time_mask.size(1), -1)), 
                    time_mask[..., None].expand((-1, -1, embs.size(1)))
                )
                node_mask = (~torch.eq(node_mapping[:, 1], -1)).to(self.device)[:, None, None]
                # Positional time encoding
                # embs = self.positional_encoding(embs, times)
                # Iterate over labels for this root node
                for label in path_dict['labels']:
                    # Only do prediction if in labels
                    if label in self.labels:
                        # Scale embeddings with alphas
                        literal_embs = literal_mapping[:, :, None].to(self.device) * embs
                        # Get alphas
                        alphas = torch.ones_like(literal_embs)
                        # Alpha embedding
                        alpha_embs = alphas * literal_embs
                        # Get root emb with ones as coefficients
                        root_emb = torch.sum(alpha_embs / time_count, dim=0)
                        # Make alpha dict for each iteration
                        alpha_dict[root_node][label] = {}
                        # Iterator over blocks
                        for block in range(self.num_blocks):
                            # Get attention query embeddings
                            # attention_embs = alpha_embs
                            attention_embs = literal_embs
                            # # Get root node for attention manipulation
                            attention_root_emb = self.normalizations_alpha_ff[label][block](root_emb)
                            # attention_root_emb = root_emb / torch.sqrt(torch.Tensor([block + 1.0])).to(self.device)
                            # Square block mask
                            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(attention_root_emb.size(0))
                            # Attend
                            x, _ =  self.attentions[label][block](
                                attention_root_emb, 
                                attention_root_emb, 
                                attention_root_emb, 
                                attn_mask=attn_mask.to(self.device)
                            )
                            # Add
                            attention_root_emb = self.normalizations_attention[label][block](attention_root_emb + x)
                            # Select last time-step
                            attention_root_emb = attention_root_emb[-1:, ...]
                            # attention_root_emb = attention_root_emb + x
                            # Upscale downscale
                            x = self.downscale[label][block](
                                torch.nn.functional.relu(
                                    self.upscale[label][block](attention_root_emb)
                                )
                            )
                            # Add
                            attention_root_emb = self.normalizations_ff[label][block](attention_root_emb + x)
                            # attention_root_emb = attention_root_emb + x
                            # Initialize heads and storage for alphas
                            concat_embs = []
                            alpha_dict[root_node][label][f'layer_{block}'] = {}
                            # Change root embedding with block
                            projected_attention_root_emb = self.attention_weights[label][block](attention_root_emb)
                            # Project literal weights
                            projected_attention_embs = self.literal_weights[label][0][head](attention_embs)
                            # Re-calculate alpha
                            alphas = self.alpha(
                                time_mask, projected_attention_embs, projected_attention_root_emb)[..., None]
                            alphas = alphas * node_mask
                            # Calculate weighted sum
                            concat_embs.append(alphas * attention_embs)
                            # Add alphas
                            alpha_dict[root_node][label][f'layer_{block}'][f'head_{0}'] = alphas
                            # Aggregate paths
                            x = torch.sum(alpha_embs / time_count, dim=0)                            
                            # Add
                            # root_emb = self.normalizations_alpha_ff[label][block](root_emb +  x)
                            root_emb = root_emb + x
                        # Make final root embedding    
                        final_root_emb = root_emb / torch.sqrt(torch.Tensor([self.num_blocks + 1.0])).to(self.device)
                        # Add output to predictions
                        output_dict[root_node][label] = final_root_emb
            # Append output
            outputs_out.append(output_dict)
            # Append alpha
            alphas_out.append(alpha_dict)
        # Return output and alpha
        return outputs_out, alphas_out
"""