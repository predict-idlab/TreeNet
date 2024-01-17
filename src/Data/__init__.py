import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from collections import deque
from typing import *

from .utils import make_numerical, read_graph


class Node:
    """Super class for EHR node type with necessary attributes for conversion."""

    def __init__(self,
                 identifier: str,
                 start: Optional[List] = None,
                 stop: Optional[List] = None,
                 values: Optional[List] = None,
                 labels: Optional[Dict] = None,
                 properties: Optional[Dict] = None,
                 neighbors: Optional[Dict] = None
                 ):
        """
        Initialize node class.

        Parameters
        ----------
        identifier: str
            Identifier for node. This is going to be used to embed the node.
        start: List[int]
            Start time stamps for node.
            (default is -inf for minimal time in graph)
        end: List[int]
            Ending time stamps for node.
            (default is +inf for infite duration)
        values: List[float]
            Value of literal.
            (Default is None.)
            (Default gets initialized to non-literal with value [1.] * len(start) for each time stamp)
        labels: dict
            Dictionary containing labels for this node.
        properties: dict
            Dictionary containing properties for this node.
            These properties are strings that are going to be embedded
        neighbors: dict[Tuple]
            Dictionary containing neighbors and the relations to these neighbors.

        Notes
        -----
        1. Start and stop times are lists to be equivalent with Literal changes.
           This should in theory not be necessary for property nodes.
        2. Properties is also list so that multiple properties can be put into node.
        """
        # Initialize
        if start is None:
            start = [-np.inf] # [0]
        if stop is None:
            stop = [np.inf] # [-1] 
        if labels is None:
            labels = {}
        if neighbors is None:
            neighbors = {}
        if properties is None:
            properties = {}
        # Identifier
        self.identifier = identifier
        # Start and end times for node
        self.start = start
        self.stop = stop
        # Assert amount of starts and stops
        assert len(self.start) == len(self.stop)
        # Make sure arrays have same length --> bad fix
        if values == None:
            self.values = [1.] * len(start)
        elif len(values) == 0:
            self.values = [1.] * len(start)
        elif len(values) < len(start):
            self.values = values + [0.0] * (len(start) - len(values))
        elif len(values) > len(start):
            self.values = values[:len(start)]
        else:
            self.values = values
        # Assert value per timestamp
        len(self.start) == len(self.values)
        # Labels for node
        self.labels = labels
        # Properties
        self.properties = properties
        # Neighbours
        self.neighbors = deepcopy(neighbors)

    # Add neighbor function
    def add_neighbor(self, relation, node):
        # Add neighbor with specific relation
        self.neighbors[(relation, node.identifier)] = node

    # Add plotting function
    def plot(self, ax):
        if ax is not None:
            ax.bar()
        else:
            plt.bar()
        a = 0
        # TODO: Fill with plotting


class GraphProcessor:
    def __init__(self, nodes_dict, relations_dict, exclusion_nodes, epsilon=None):
        # Get node and relations dict
        self.nodes_dict = nodes_dict
        self.relations_dict = relations_dict
        # Exclusion node
        self.exclusion_nodes = exclusion_nodes
        # set epsilon parameter
        self.epsilon =epsilon
        # reset identifiers & graph structures on creation
        self.reset()

    def node_indices(self):
        self.i = 0
        while True:
            yield self.i
            self.i += 1

    def relation_indices(self):
        self.j = 0
        while True:
            yield self.j
            self.j += 1

    def reset(self):
        # Reset node and relation indices
        self._node_indices = self.node_indices()
        self._relation_indices = self.relation_indices()
        # Initialize node mapping
        self.node_mapping = []
        # Initialize relation mapping
        self.relation_mapping = []
        # Initialize literal values
        self.literal_values = []
        # Initialize time mapping
        self.time_mapping = []
        # Initialize edge index
        self.edge_index = []
        # Initialize edge type
        self.edge_type = []
        # Initialize label mapping
        self.label_mapping = {}

    def get_relation(self, relation, mode):
        if mode == 'train':
            if relation not in self.relations_dict:
                self.relations_dict[relation] = len(self.relations_dict) * 2
        return self.relations_dict.get(relation, -1)

    def exclude_node(self, value):
        return ~any([node in str(value) for node in self.exclusion_nodes])

    def get_node(self, value, mode):
        if mode == 'train':
            if (value not in self.nodes_dict) & (self.exclude_node(value)):
                self.nodes_dict[value] = len(self.nodes_dict)
        return self.nodes_dict.get(value, -1)

    def map_node(self, node, index=None, mode='train'):
        # Get node index and identifier
        node_index = next(self._node_indices) if index is None else index
        # Get identifier
        node_identifier = self.get_node(node.identifier, mode)
        # Add to node mapping
        self.node_mapping.append([node_index, node_identifier])
        # Append literal value
        self.literal_values.append(node.values)
        # Append timestamps
        self.time_mapping.append(list(zip(node.start, node.stop)))
        # Add properties
        if not len(node.properties) == 0: ##
            for relation, value in node.properties.items():
                # Make node
                value_index = next(self._node_indices)
                # Get node for value
                value_identifier = self.get_node(value, mode)
                # Add top node mapping
                self.node_mapping.append([value_index, value_identifier])
                # Get relation index
                relation_idx = next(self._relation_indices)
                # Make relation
                relation_identifier = self.get_relation(relation, mode)
                # Append relation type and node to edge index and type
                self.edge_index.append([node_index, value_index])
                self.edge_type.append(relation_identifier)
                # Add to relation mapping
                self.relation_mapping.append([relation_idx, relation_identifier])
                # Append literal value
                self.literal_values.append(node.values)
                # Append timestamps
                self.time_mapping.append(list(zip(node.start, node.stop)))

        if not len(node.labels) == 0:
            # Add root node to label mapping
            self.label_mapping[node_index] = {}
            # for each label add corresponding information
            for label, value in node.labels.items():
                # Initialize label mapping if needed
                label_dict = {
                    'labels': [],
                    'start': [],
                    'stop': []
                }
                # Set value using extend
                value = value if isinstance(value, list) else [value]
                label_dict['labels'].extend(value)
                # Set times
                label_dict['start'].extend(node.start)
                label_dict['stop'].extend(node.stop)
                # Add label dict to root node
                self.label_mapping[node_index][label] = label_dict

    def unpack(self, node, pairs, rels):
        # pairs and rels are maintained for compatibility
        pairs = []
        rels = []
        Q = deque([node])
        poped_nodes = set()
        while(len(Q) != 0):
            node = Q.pop()
            x = node.neighbors.items()
            for (rel, idx), value in x:
                # Append node pair
                pairs.append([node, value])
                rels.append(rel)
                if value not in poped_nodes and value not in Q:
                    Q.append(value)
            poped_nodes.add(node)
        return pairs, rels
    
    # def unpack(self, node, pairs, rels):
    #     if not len(node.neighbors) == 0:
    #         for (rel, idx), value in node.neighbors.items():
    #             # Append node pair
    #             pairs.append([node, value])
    #             rels.append(rel)
    #             # unpack further
    #             self.unpack(value, pairs, rels)
    #     return pairs, rels

    # def unpack(self, node):
    #     pairs = []
    #     rels = []
    #     Q = deque([node])
    #     poped_nodes = set()
    #     while(len(Q) != 0):
    #         node = Q.pop()
    #         x = node.neighbors.items()
    #         for (rel, idx), value in x:
    #             # Append node pair
    #             pairs.append([node, value])
    #             rels.append(rel)
    #             if value not in poped_nodes and value not in Q:
    #                 Q.append(value)
    #         poped_nodes.add(node)
    #     return pairs, rels
    
    def transform_dataset(self, dataset, reset=True, mode='train', add_time=True):
        for patient in tqdm(dataset):
            data = self.transform(patient.graph, reset, mode, add_time)
            for key, value in data.items():
                setattr(patient, key, value)
        return dataset

    def transform(self, patient, reset=True, mode='train', add_time=False):
        # Reset identifiers & graph structures
        if reset:
            self.reset()

        # Unpack neighborshoods
        pairs, rels = self.unpack(patient, [], [])
        # Get all unique nodes
        unique_nodes = list(set([i for pair in pairs for i in pair]))
                
        # Give node indices to unique nodes
        node_idx = [next(self._node_indices) for node in unique_nodes]
        # Map all pairs into edge index and edge type
        for pair, relation in zip(pairs, rels):
            # Add edge indices to edge index
            edge_idx = [node_idx[unique_nodes.index(pair[0])], node_idx[unique_nodes.index(pair[1])]]
            self.edge_index.append(edge_idx)
            # Get relation index
            relation_idx = next(self._relation_indices)
            # Get relation type
            relation_identifier = self.get_relation(relation, mode)
            # Add to edge type
            self.edge_type.append(relation_identifier)
            # Add to relation mapping
            self.relation_mapping.append([relation_idx, relation_identifier])

        # Map all nodes in this unpacked neighborhood
        for index, node in zip(node_idx, unique_nodes):
            self.map_node(node, index, mode)
            
        times, out, mask = self.map_values_times(self.literal_values, self.time_mapping, add=add_time)
        # dataset.node_mapping = torch.tensor(self.node_mapping, dtype=torch.long)
        # dataset.relation_mapping = torch.tensor(self.relation_mapping, dtype=torch.long)
        # Make edge index
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_type = torch.tensor(self.edge_type, dtype=torch.long)
        # return dataset
        return {
            'node_mapping': torch.tensor(self.node_mapping, dtype=torch.long),
            'relation_mapping': torch.tensor(self.relation_mapping, dtype=torch.long),
            'labels': self.label_mapping,
            'literal_mapping': torch.tensor(out, dtype=torch.float),
            'times': times,
            'time_mapping': self.time_mapping,
            'time_mask': mask,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'node_degree': edge_index.flatten().bincount()
        }

    @staticmethod
    def add_end(start_end, add, epsilon):
        # Get spacing
        spacing = 1.0 if epsilon is None else epsilon/2
        # Make tuples
        start_end = list(start_end)
        # Return added tuple when needed
        if add:
            start_end += [start_end[-1] + spacing]
        return start_end

    def map_values_times(self,
                         literal_values,
                         time_mapping,
                         add: bool = False,
                         ):
        
        epsilon = self.epsilon
        # check time lengths
        len_values = [len(v) for v in literal_values]
        len_times = [len(t) for t in time_mapping]
        # Initialize
        unpacked_times = []
        unpacked_values = []
        indices = []
        # Unpack values and times while stroing indices
        for idx, (times, values) in enumerate(zip(time_mapping, literal_values)):
            n_times = len(times)
            n_values = len(values)
            if n_times > n_values:
                values.extend([0.0] * (n_times - n_values))
            elif n_times < n_values:
                n_times.extend([(-np.inf, np.inf)] * (n_values - n_times))
            for time, value in zip(times, values):
                unpacked_times.append(time)
                unpacked_values.append(make_numerical(value)) # -> maps strings to 0.0. Fix in graph making.
                indices.append(idx) 
                
        # self.add_end(start_end, add, epsilon)
            
        # Flatten values array
        literal_values = np.array([unpacked_values]).T 
        # Make indices
        indices = np.array([idx for i, node_times in enumerate(time_mapping) for idx in [i] * len(node_times)])
      
        # Get number of nodes
        num_nodes = len(time_mapping)
        
        # Get all unique times in array
        times = np.unique(unpacked_times)
        # Remove infinities from unique times
        times = np.array([time for time in times if time not in [-np.inf, np.inf]])
        # Failsafe for empty array
        if len(times) == 0:
            times = np.array([0])
        # Maximum time
        max_time = np.max(times)
        # Minimum time
        min_time = np.min(times)
        # If epsilon is None do nothing
        if epsilon is None:
            # Spacing parameter
            epsilon = 10e-4
        else:
            # Get equally spaced time lattice
            times = np.arange(min_time, max_time, epsilon)
        # Flatten time array
        time_mapping = np.array([start_end for node_times in time_mapping for start_end in node_times])
        # Another failsafe
        if len(time_mapping.shape) != 2:
            time_mapping = np.array([(0, 0)])
        # Change +-infinity to minimal and maximal time respectively
        time_mapping[time_mapping == -np.inf] = min_time
        time_mapping[time_mapping == np.inf] = max_time
        # Convert to boolean array
        bools = np.stack([times >= time_mapping[:, 0:1], times < time_mapping[:, 1:]], axis=-1)
        # Check nodes that are in time intervals
        value_mask = np.all(bools, axis=-1)
        # Create value mask using literal values
        value_mapping = np.where(value_mask, literal_values, 0)
        # Index into node length array using indices
        out = np.zeros((num_nodes, times.size))
        # Make output mask
        out_mask = [[] for _ in range(num_nodes)]
        # Iterate over indices
        for i, j in enumerate(indices):
            # Get values
            values =  value_mapping[i, ...]
            # Get mask
            mask = value_mask[i, ...]
            # Sum values
            out[j, ...] += values
            # Append mask
            out_mask[j].append(mask)
        # Check time steps where node is not active based on initial mask
        out_mask = np.array([np.any(np.stack(m, axis=0), axis=0) for m in out_mask])
        # return times and mask
        return times, out, out_mask


class PathBuilder:
    """Build paths from patient graphs for input into TreePathRGCN."""
    def __init__(self,
                 max_depth: int = 10,
                 max_new_pairs: int = 20,
                 min_multiplication: int = 2,
                 delimiter: str = '.',
                 verbose: bool = True
                 ):
        self.max_depth = max_depth
        self.max_new_pairs = max_new_pairs
        self.min_multiplication = min_multiplication
        self.delimiter = delimiter
        self.verbose = verbose

    @staticmethod
    def get_pairs(elements):
        return [pair for pair in list(zip(elements, elements[1:]))]

    def reduce_paths(self, paths: list):
        # Initialize dict for new pairs, counter
        new_pairs = {}
        counter = 0
        # While reduction is possible create new pairs
        for it in range(self.max_new_pairs):
            # Get pairs of relations in long list
            pairs = [
                self.delimiter.join(list(pair))
                for path in paths
                for pair in self.get_pairs(path.split(self.delimiter)) if len(pair) == 2
            ]
            # Count unique pairs
            unique_pairs, counts = np.unique(pairs, return_counts=True)
            if counts.size == 0:
                break
            # Get pair that occurs the most
            max_pair = unique_pairs[counts.argmax()]
            # Make new pair and count in negatives
            new_pair = str(-counter)
            # Store new pair
            new_pairs[new_pair] = max_pair
            ### TODO: MIGHT CAUSE ISSUES TRY WITH TUPLE REPLACEMENT IF NOT WORKING '20.20.' '20.2.' -> '-1' '-10'
            ### TODO: THIS NEEDS TO BE CLEANER AND FASTER
            # Replace maximum pair with new pair     
            max_pair_ = self.delimiter + max_pair + self.delimiter
            new_pair_ = self.delimiter + new_pair + self.delimiter
            paths = [
                (self.delimiter + path + self.delimiter).replace(max_pair_, new_pair_).replace('--', '-')[len(self.delimiter):-len(self.delimiter)]
                for path in paths
            ]
            # Increment counter 
            counter += 1
            # Check feasibility: 
            feasible = counts.max() > self.min_multiplication
            # break when counts is smaller than minimal multiplications
            if not feasible:
                break 
        # Map back onto integers
        paths = [list(map(int, path.split(self.delimiter))) if path != '' else [] for path in paths]
        # new_pairs
        new_pairs = {int(key): tuple(map(int, value.split(self.delimiter))) for key, value in new_pairs.items()}
        # return paths and new pairs dict
        return paths, new_pairs

    def calculate_paths(self, edge_index, relation_mapping, root_node):
        # Node degree
        node_degree = edge_index.flatten().bincount()
        # edges
        edge_type = relation_mapping[:, -1]
        edge_indices = relation_mapping[:, 0]
        # calculate num nodes
        num_nodes = edge_index.max() + 1
        # Set stop condition
        condition = True
        # Iteration counter
        iteration = 0
        # Get paths list
        paths = [[] for i in range(num_nodes)]
        edges = [[] for i in range(num_nodes)]
        degrees = [[] for i in range(num_nodes)]
        # Initialize nodes as root node
        nodes = torch.tensor([root_node])
        # Initialize previous nodes
        prev_nodes = torch.tensor([])
        # Iterate through tree
        while condition:
            new_nodes = []
            # Initialize path for iteration
            for node in nodes:
                # Mask for connection relating to node
                mask = edge_index == node
                # Get out nodes for this node
                out_nodes = edge_index[mask[:, 0], 1]
                out_edges = edge_indices[mask[:, 0]]
                out_types = edge_type[mask[:, 0]] + 1
                # Get in nodes for this node
                in_nodes = edge_index[mask[:, -1], 0]
                in_edges = edge_indices[mask[:, -1]]
                in_types = edge_type[mask[:, -1]]
                # Check if in previous nodes
                in_mask = torch.isin(in_nodes, prev_nodes)
                in_nodes = in_nodes[~in_mask]
                in_edges = in_edges[~in_mask]
                in_types = in_types[~in_mask]
                out_mask = torch.isin(out_nodes, prev_nodes)
                out_nodes = out_nodes[~out_mask]
                out_edges = out_edges[~out_mask]
                out_types = out_types[~out_mask]
                # Add to paths
                for n, r, t in zip(out_nodes, out_edges, out_types):
                    degrees[n].append(node_degree[node])
                    paths[n].extend(paths[node])
                    paths[n].append(r.item())  
                    edges[n].append(t.item())
                for n, r, t in zip(in_nodes, in_edges, in_types):
                    degrees[n].append(node_degree[node])
                    paths[n].extend(paths[node])
                    paths[n].append(r.item())
                    edges[n].append(t.item())
                # Append new nodes
                new_nodes.append(torch.concat([in_nodes, out_nodes]))
            # Stop when no more nodes available
            if len(new_nodes) == 0:
                condition = False
            else:
                prev_nodes = nodes
                nodes = torch.unique(torch.concat(new_nodes, dim=0))
            # Increment iteration
            iteration += 1
            # Check condition
            condition = iteration < self.max_depth
        # Convert to string with delimiter
        paths = [self.delimiter.join([str(p) for p in path]) for path in paths]
        return paths, edges, degrees
    
    def transform_dataset(self, dataset):
        iterator = dataset
        if self.verbose:
            iterator = tqdm(dataset)
        for data in iterator:
            paths = self.transform(data.edge_index, data.relation_mapping, data.labels)
            setattr(data, 'paths', paths)
        return dataset

    def transform(self, edge_index, relation_mapping, label_dict: List):
        """Build trees based on root nodes.

        Parameters
        ----------
        data: dict
            Data dictionary from tree processor
        mode: str
            'train', 'val' or 'test' mode.
        root_nodes: Optional[list[dict]]
            List of root nodes.
            similar structure as labels from train mode, without actual labels
            --> see notes in TreeProcessor.
        Returns
        -------
        dict:
            Data dictionary
        """
        paths_dict = {}
        pairs_dict = {}
        for root, labels in label_dict.items():
            # labels = {key: value for key, value in labels.items() if key == 'has_expired'}
            # if len(labels) != 0:
            # Calculate paths
            paths_list, edges_list, degrees = self.calculate_paths(edge_index, relation_mapping, root)
            # Reduce paths with new relations
            paths_list, new_pairs_dict = self.reduce_paths(paths_list)
            # Add to dict
            paths_dict[root] = {
                'paths': paths_list, 
                'edges': edges_list,
                'pairs': new_pairs_dict,
                'labels': list(labels.keys()),
                'degrees': degrees
            }
            
        # Return paths
        return paths_dict


class Data:
    def __init__(self, name, graph):
        self.name = name
        self.graph = graph


# Make a dataset
class TreeDataset(Dataset):
    def __init__(
            self,
            graph_dir,
            sample_names: Optional[list[int]] = None
    ):
        # Set graph directory
        self.graph_dir = graph_dir
        # Get sample names
        self.sample_names = sample_names if sample_names is not None else os.listdir(graph_dir)
        self.samples = [Data(sample_name, read_graph(os.path.join(self.graph_dir, sample_name))) for sample_name in
                        sample_names]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TreeDataModule():
    def __init__(self, graph_dir: str, samples: list, batch_size: int, processor: GraphProcessor):
        super().__init__()
        self.graph_dir = graph_dir
        self.sample_names = samples
        self.batch_size = batch_size
        self.processor = processor
        self.path_builder = PathBuilder()

    @staticmethod
    def remove_nans(dataset):
        for data in dataset:
            value = data.literal_mapping
            value[torch.isnan(value)] = 0.0
            data.literal_mapping = value
        return dataset

    def transform(self, dataset, mode):
        print('Process graphs')
        # Process dataset
        dataset = self.processor.transform_dataset(dataset, True, mode, False)
        # Build paths
        print('Build paths')
        dataset = self.path_builder.transform_dataset(dataset)
        # Remove nans
        print('Remove nans')
        dataset = self.remove_nans(dataset)
        # Return dataset
        return dataset

    # Batch dataset function
    def batch(self, dataset):
        return [dataset[i: i + self.batch_size] for i in range(0, len(dataset), self.batch_size)]

    def setup(self, stage=None):
        # Make datasets
        self.dataset = TreeDataset(self.graph_dir, self.sample_names)
        # Transform datasets
        self.dataset = self.transform(self.dataset, stage)


    def dataloader(self):
        batched_dataset = self.batch(self.dataset)
        return DataLoader(batched_dataset, batch_size=None, batch_sampler=None)

"""
LEGACY CODE
"""
# TODO: Find better location for storage

'''
class TreeBuilder:
    """Build trees from patient graphs for input into TreeRGCN."""
    def __init__(self, max_depth: int = 16, max_branch: int = 1, verbose: bool = True):
        self.max_depth = max_depth
        self.max_branch = max_branch
        self.verbose = verbose

    def max_branched_edges_per_node(
            self,
            _node: torch.tensor,
            level_edge: torch.tensor,
            level_type: torch.tensor
    ) -> torch.tensor:
        """Calculate branching actor and mask

        Parameters
        ----------
        _node: tensor
            single item tensor containing node type
        level_edge: tensor
            edge index tensor for this level
        level_type
            edge type tensor for this level
        Returns
        -------
            indices of nodes with branching factor under the max branching factor
        """
        # Mask per incoming node
        node_mask = _node == level_edge[:, 0]
        # Make mask full of true
        indices = torch.arange(level_edge.size(0))
        # Calculate counts per relation for node
        _type, counts = torch.unique(level_type[node_mask], return_counts=True)
        # Mask relations that have incoming node and are of type that branches too much
        return indices[node_mask][torch.isin(level_type[node_mask], _type[counts < self.max_branch])]

    def build_tree(
            self,
            root_node: Union[int, torch.tensor],
            edge_index: torch.tensor,
            edge_type: torch.tensor
    ) -> List[List[torch.tensor]]:
        """Build tree from edge index representation.

        Parameters
        ----------
        root_node: tensor or int
            Tensor containing root node or int specifying the root node
        edge_index: tensor
            edge index tensor
        edge_type: tensor
            edge type tensor
        verbose: bool
            Verbosity bool
        Returns
        -------
        List:
            Tree structure for root node from edge index representation.
        """
        # TODO: add condition that ends loop when no more leaf nodes are available.
        tree = []
        if isinstance(root_node, torch.Tensor):
            assert root_node.size(0) == 1 & len(root_node.shape) == 1 \
                , "When passing root node tensor should be single element tensor with shape (1, )."
            # Make root node the correct variable
            root_nodes = root_node
        elif isinstance(root_node, int):
            # Single root node at base to start
            root_nodes = torch.tensor([root_node])
        else:
            raise ValueError('root_node must be tensor or int.')
        # Initialize original index counter for edges
        indices = torch.arange(edge_index.size(0))
        # Iterate over max tree depth
        for level in range(self.max_depth):
            # When tree doesn't reach max depth break
            if indices.size(0) == 0:
                break
            # When tree doesn't reach max depth break
            if root_nodes.size(0) == 0:
                break
            # Mask for edges from root to branch
            root_mask = torch.isin(edge_index[:, 0], root_nodes)
            # Corresponding edges
            level_edge = edge_index[root_mask, :]
            # Corresponding edge types
            level_type = edge_type[root_mask]
            # Mask edge index counter
            level_idx = indices[root_mask]
            # If level is empty break
            if level_edge.size(0) == 0:
                break
            # Make mask for relations that have to many branches incoming
            max_branch_idx = torch.cat([
                self.max_branched_edges_per_node(_node, level_edge, level_type)
                for _node in torch.unique(root_nodes)
            ], dim=0)
            # Mask edges, types and index counter
            level_edge = level_edge[max_branch_idx, :]
            level_type = level_type[max_branch_idx]
            level_idx = level_idx[max_branch_idx]
            # Remove indices from original arrays to avoid recounting
            indices = indices[~root_mask]
            # Select remaining edges and types based on previous index counter
            edge_index = edge_index[~root_mask, :]
            edge_type = edge_type[~root_mask]
            # Overwrite root nodes
            root_nodes = level_edge[:, 1]
            # Append level to tree
            tree.append([level_edge, level_type, level_idx])
        return tree

    def transform(self, data, mode: str = 'train', root_nodes: Optional[List[dict]] = None):
        """Build trees based on root nodes.

        Parameters
        ----------
        data: dict
            Data dictionary from tree processor
        mode: str
            'train', 'val' or 'test' mode.
        root_nodes: Optional[list[dict]]
            List of dictionaries containing labels and corresponding root nodes for test mode.
            similar structure as labels from train mode, without actual labels
            --> see notes in TreeProcessor.
        Returns
        -------
        dict:
            Data dictionary
        """
        assert mode in ['train', 'val', 'test'] \
            , "mode must be 'train', 'val' or 'test'."
        if mode in ['train', 'val']:
            assert 'labels' in data.keys() \
                , "Labels must be in data when mode is 'train' or 'val'."
            root_nodes = [
                {
                    label: label_dict['root_nodes']
                    for label, label_dict in labels.items()
                }
                for labels in data['labels']
            ]
        else:
            assert root_nodes is not None \
                , "Root nodes must be given in mode 'test'."
        # Iterate over root nodes, edge index and edge type in data
        iterator = zip(root_nodes, data['edge_index'], data['edge_type'])
        if self.verbose:
            iterator = tqdm(iterator)
        # Build trees for each graph for all root nodes and labels
        trees = [
            {
                label: [self.build_tree(node.item(), edge_index, edge_type) for node in root_node]
                for label, root_node in labels.items()
            }
            for labels, edge_index, edge_type in iterator
        ]
        # Add trees to dict
        data['tree'] = trees
        return data


class DataLoader:
    """Data loader class for pipeline."""
    def __init__(self, data_directory: str, verbose: bool = True):
        """Initialize data loader object.

        Parameters
        ----------
        data_directory: str
            directory for data files
        """
        self.data_dir = data_directory
        self.verbose = verbose

    def load(self, patients: list) -> list:
        """Load patient data.

        Parameters
        ----------
        patients: list[str]
            List of patient file to load

        Returns
        -------
        list[dict]:
            List of patient graph dictionaries
        """
        if self.verbose:
            iterator = tqdm(patients)
        else:
            iterator = patients
        return [self.load_patient(patient) for patient in iterator]

    def load_patient(self, patient: str) -> dict:
        """Load all patient data

        Parameters
        ----------
        patient: str
            Patient identifier

        Returns
        -------
        dict
            Dictionary containing graph and mappings
        """
        # Get node and relation to index mappings
        idx_to_node, rel_to_idx = self.load_mapping(patient)
        # Get edge indices and types
        edge_index, edge_type = self.load_graph(patient)
        return {
            'idx_to_node': idx_to_node,
            'rel_to_idx': rel_to_idx,
            'edge_index': edge_index,
            'edge_type': edge_type
        }

    def load_graph(self, patient: str) -> Tuple[torch.tensor, torch.tensor]:
        """Load edge index and edge type graph representation.

        Parameters
        ----------
        patient: str
            String identifier for patient

        Returns
        -------
        array x2:
            Edge index and edge type
        """
        with open(join(self.data_dir, patient, 'edge_index.txt'), 'rb') as f:
            edge_index = loadtxt(f, dtype=int32, delimiter=',')
        with open(join(self.data_dir, patient, 'edge_type.txt'), 'rb') as f:
            edge_type = loadtxt(f, dtype=int32, delimiter=',')
        return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_type, dtype=torch.long)

    def load_mapping(self, patient: str) -> Tuple[dict, dict]:
        """Load node and relation mappings from rdf to integer.

        Parameters
        ----------
        patient: str
            Patient identifier

        Returns
        -------
        dict x2:
            integer index of node to rdf representation 'idx_to_node'.
            relation rdf to integer 'rel_to_idx'.
        """
        with open(join(self.data_dir, patient, 'idx_to_node.pkl'), 'rb') as f:
            idx_to_node = load(f)
        with open(join(self.data_dir, patient, 'rel_to_idx.pkl'), 'rb') as f:
            rel_to_idx = load(f)
        idx_to_node = {int(idx): node for idx, node in idx_to_node.items()}
        rel_to_idx = {rel: int(idx) for rel, idx in rel_to_idx.items()}
        return idx_to_node, rel_to_idx


class TreeProcessor:
    """Tree processor class"""
    def __init__(
            self,
            labels: list,
            verbose: bool = True,
            exclusion_relations: Optional[list] = None,
            exclusion_nodes: Optional[list] = None,
            unk: int = -1,
            delimiter: str = '+'
    ):
        self.labels = labels
        self.verbose = verbose
        self.exclusion_nodes = exclusion_nodes
        self.exclusion_relations = exclusion_relations
        self.unk = unk
        self.delimiter = delimiter

        self.nodes_dict = None
        self.literals_dict = None
        self.relations_dict = None

    def fit(self, patient_dicts, reset: bool = True):
        """build tree processor on train patients

        Parameters
        ----------
        patient_dicts: dict
            patient dicts from data loader
        reset: bool
            Whether to reset sets containing rdf variables or keep building on existing fit.
        """
        if reset:
            nodes = set()
            relations = set()
            literals = set()
        else:
            assert None not in [self.nodes_dict, self.literals_dict, self.relations_dict], \
                f'Processor must be built on train patients when reset is {reset}.'
            nodes = set(self.nodes_dict.keys())
            relations = set(self.nodes_dict.keys())
            literals = set(self.nodes_dict.keys())
        iterator = patient_dicts
        if self.verbose:
            iterator = tqdm(iterator)
        for data in iterator:
            node, literal = self.filter(data['idx_to_node'])
            nodes.update(node.values())
            for lit in literal.values():
                datatype = lit.datatype if lit.datatype is not None else 'None'
                literals.update(self.split_literal(datatype))
            relations.update(data['rel_to_idx'].values())

        if self.exclusion_nodes is not None:
            nodes = self.exclude(nodes, self.exclusion_nodes)
        if self.exclusion_relations is not None:
            relations = self.exclude(relations, self.exclusion_relations)

        self.nodes_dict = {node: idx for idx, node in enumerate(nodes)}
        self.literals_dict = {literal: idx for idx, literal in enumerate(literals)}
        self.relations_dict = {relation: idx * 2 for idx, relation in enumerate(relations)}
        
    def split_literal(self, literal):
        return literal.split(self.delimiter)

    def get_dicts(self):
        """Get dictionaries after fitting.

        Returns
        -------
        dict x3
            nodes, literals and relations dicts
        """
        assert None not in [self.nodes_dict, self.literals_dict, self.relations_dict], \
            'Processor must be built on train patients.'
        return self.nodes_dict, self.literals_dict, self.relations_dict

    def transform(self, patient_dicts, mode: str = 'train'):
        """Transform data into graph mapping and edge representation.

        Parameters
        ----------
        patient_dicts: list
            List of patient dictionaries
        mode: str
            Mode for processing, must be in 'train', val' or 'test'.
            Train and val will extract labels and add to dictionary.
            Defaults to 'train'
        Returns
        -------
        dict:
            Data dictionary. See notes for structure

        Notes
        -----
        Data dictionary structure
        - Node mapping:
            List with mapping between tree index and index in nodes dict from tree processor per patient.
        - Literal mapping:
            List with mapping between tree index and index in literals dict from tree processor per patient.
        - Relation mapping:
            List with mapping between relation index in tree and in relations dict from tree processor.
        - Literal values:
            List with values corresponding to the literals in the literal mapping.
        - Edge index:
            List with edge index graph representations per patient.
        - Edge type:
            List with edge types for each edge index tensor.
        - Labels:
            List of dictionaries containing the label information when mode is val or train.
            Each label dictionary has the following structure.
            {
                label: {
                            'root_nodes': list,
                            'labels': list
                        }
                for label in labels
            }
            Each specified label relation in self.labels gets a dictionary entry.
            This dictionary contains the following.
            root_nodes:
                The indices for all root nodes of the specified label relationships
            labels:
                The literals corresponding to each root index.
        """
        assert None not in [self.nodes_dict, self.literals_dict, self.relations_dict], \
            'Processor must be built on train patients.'
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val' or 'test'"

        # Make data dict
        data = {
            'node_mappings': [],
            'literal_mappings': [],
            'relation_mappings': [],
            'literal_values': [],
            'edge_index': [],
            'edge_type': []
        }
        if mode in ['train', 'val']:
            data['labels'] = []
        iterator = patient_dicts
        if self.verbose:
            iterator = tqdm(iterator)
        for patient_dict in iterator:
            # Get node and relation to index mappings
            idx_to_node, rel_to_idx, edge_index, edge_type = patient_dict.values()
            # Filter nodes into nodes and literals
            idx_to_node, idx_to_literal = self.filter(idx_to_node)
            # Get labels if in train or validation mode
            if mode in ['train', 'val']:
                edge_index, edge_type, label_idx = self.get_labels(rel_to_idx, edge_index, edge_type)
                # get label information
                label_dict, idx_to_literal = self.get_label_info(label_idx, idx_to_node, idx_to_literal)
                # Add labels
                data['labels'].append(label_dict)
            # Remove unwanted relations
            edge_index, edge_type = self.remove_relations(rel_to_idx, edge_index, edge_type)
            # Get mapping between nodes in patient and global dictionaries
            node_mapping = self.process_nodes(idx_to_node)
            # Get mapping between literals in patient and global dictionaries
            literal_mapping, literal_value = self.process_literals(idx_to_literal)
            # Get mapping between relations in patient and global dictionaries
            relation_mapping = self.process_relations(rel_to_idx)
            # Add values to data dictionary
            data['node_mappings'].append(node_mapping)
            data['literal_mappings'].append(literal_mapping)
            data['relation_mappings'].append(relation_mapping)
            data['literal_values'].append(literal_value)
            data['edge_index'].append(edge_index)
            data['edge_type'].append(edge_type)
        return data

    @staticmethod
    def get_label_info(label_idx, idx_to_node, idx_to_literal):
        label_dict = {}
        idx_to_node = {**idx_to_node, **idx_to_literal}
        for label, idx in label_idx.items():
            root_nodes, leaf_nodes = torch.unbind(idx, dim=-1)
            label_dict[label] = {
                'root_nodes': list(root_nodes),
                'labels': [idx_to_literal.pop(i.item()) for i in leaf_nodes]
            }
        return label_dict, idx_to_literal

    @staticmethod
    def exclude(nodes: Iterable, terms: Iterable):
        """Exclude nodes which contain any term in terms.

        Parameters
        ----------
        nodes: Iterable[Node]
            List of nodes.
        terms: Iterable[str]
            List of exclusion terms.

        Returns
        -------
        list[Node]:
            List of nodes that do not contain any of the terms.
        """
        return [node for node in nodes if not any([term in str(node) for term in terms])]

    @staticmethod
    def filter(idx_to_node: dict):
        """Split index to node dictionary in literals and nodes.

        Parameters
        ----------
        idx_to_node: dict
            Dictionary containing all nodes, both URIRef and Literal.

        Returns
        -------
        dict x2:
            idx_to_node dictionary containing all URIRefs and idx_to_literal containing all literals.
        """
        idx_to_literal = {
            idx: node for idx, node in idx_to_node.items() if isinstance(node, rdf.Literal)}
        idx_to_node = {
            idx: node for idx, node in idx_to_node.items() if not isinstance(node, rdf.Literal)
        }
        return idx_to_node, idx_to_literal

    def process_nodes(self, idx_to_node: dict) -> torch.tensor:
        """Makes node mapping from index to node dictionary and overall node dictionary.

        Parameters
        ----------
        idx_to_node: dict
            Dictionary mapping tree indices to nodes
        Returns
        -------
        torch.tensor: (N x 2)
            Tensor mapping tree indices to node indices in embedding matrix
        Notes
        -----
        idx_to_node should only contain nodes, no literals.
        Splitting can be done using the filter function.
        """
        assert all([isinstance(node, rdf.URIRef) for node in idx_to_node.values()]), \
            "all 'idx_to_node' values should be rdf nodes."
        node_mapping = [[idx, self.nodes_dict.get(node, self.unk)] for idx, node in idx_to_node.items()]
        return torch.tensor(node_mapping, dtype=torch.long)

    def process_relations(self, relations_to_idx: dict) -> torch.tensor:
        """Makes relation mapping from index to relation dictionary and overall relation dictionary.

        Parameters
        ----------
        relations_to_idx: dict
            Dictionary mapping relation indices to relations

        Returns
        -------
        torch.tensor: (R x 2)
            Tensor mapping relation indices to relation indices in embedding matrix.
        """
        relation_mapping = [[idx, self.relations_dict.get(rel, self.unk)] for rel, idx in relations_to_idx.items()]
        return torch.tensor(relation_mapping, dtype=torch.long)

    def process_literals(self, idx_to_literal: dict) -> torch.tensor:
        """Makes literal mapping from index to literal dictionary and overall literal dictionary.

        Parameters
        ----------
        idx_to_literal: dict
            Dictionary mapping literal indices to literals

        Returns
        -------
        torch.tensor x2: (L x 2) x (L, )
            Tensor mapping literal indices to literal indices in embedding matrix and corresponding literal value.

        Notes
        -----
        idx_to_literal should only contain literals.
        Splitting can be done using the filter function.
        """
        assert all([isinstance(node, rdf.Literal) for node in idx_to_literal.values()]), \
            "all 'idx_to_literal' values should be rdf literals."
        literal_mapping = []
        literal_values = []
        for idx, lit in idx_to_literal.items():
            datatype = lit.datatype if lit.datatype is not None else 'None'
            for part in self.split_literal(datatype):
                literal_mapping.append([idx, self.literals_dict.get(part, self.unk)])
                literal_values.append(make_numerical(lit))
        return torch.tensor(literal_mapping, dtype=torch.long), torch.tensor(literal_values, dtype=torch.float)

    def get_labels(self, relations_to_idx, edge_index, edge_type) -> Union[torch.tensor, torch.tensor, dict]:
        """Get indices corresponding to leaf nodes of specified label relations.

        Parameters
        ----------
        relations_to_idx: dict
            relation to index dict
        edge_index: torch.tensor
            edge index tensor
        edge_type: torch.tensor
            edge type tensor

        Returns
        -------
        torch.tensor x2, dict:
            Updated edge index and edge type tensors with specified relations removed
            with the corresponding indices of label nodes per label type.
        """
        idx = {}
        for label in self.labels:
            rel_idx = relations_to_idx[label]
            mask = edge_type == rel_idx
            idx[str(label)] = edge_index[mask, :]
            edge_index, edge_type = edge_index[~mask], edge_type[~mask]
        return edge_index, edge_type, idx

    def remove_relations(self, relations_to_idx, edge_index, edge_type):
        """Remove relationships define in 'relations' from graph.

        Parameters
        ----------
        relations_to_idx: dict
            Relationship to index dictionary
        edge_index: torch.tensor
            edge index tensor
        edge_type: torch.tensor
            edge type tensor

        Returns
        -------
        torch.tensor x2:
            Updated edge index and edge type tensors with specified relations removed.
        """
        for relation in self.exclusion_relations:
            rel_idx = relations_to_idx[relation]
            mask = edge_type == rel_idx
            edge_index, edge_type = edge_index[~mask], edge_type[~mask]
        return edge_index, edge_type


class LiteralProcessor:
    """Class to process literals."""
    def __init__(self, normalizations, verbose: bool = True):
        """Initialize processor

        Parameters
        ----------
        normalizations: dict
            dict of normalizations that map literal to normalization procedure
            TODO: Add functionality, currently hard coded standard normalization
        verbose: bool
            Verbosity boolean
        """
        self.normalizations = normalizations
        self.verbose = verbose
        # Initialize statistics as None
        self._stats = None

    @staticmethod
    def get_stats(tensor: torch.tensor) -> dict:
        """Get statistics from tensor

        Parameters
        ----------
        tensor: torch.tensor

        Returns
        -------
        dict:
            All statistics needed for normalization
        """
        return {
            'min': tensor.min(),
            'max': tensor.max(),
            'mean': tensor.mean(),
            'std': tensor.std()
        }

    def fit(self, data: dict):
        """Fit processor.

        Parameters
        ----------
        data: dict
            data dictionary as gotten from TreeProcessor
        """
        # Unpack dictionary
        mappings, values = data['literal_mappings'], data['literal_values']
        # Values dictionary
        values_dict = {}
        # Make iterator
        iterator = zip(mappings, values)
        if self.verbose:
            tqdm(iterator)
        # concatenate all data
        for mapping, values_ in iterator:
            nodes = mapping[:, 0]
            types = mapping[:, 1]
            for node in torch.unique(nodes):
                mask = nodes == node
                key = '+'.join([str(t) for t in types[mask].tolist()])
                value = values_[mask].mean()
                if key not in values_dict.keys():
                    values_dict[key] = []
                values_dict[key].append(value)
        # Get statistics per literal type
        self._stats = {
            key: self.get_stats(torch.tensor(values))
            for key, values in values_dict.items()
        }

    def transform(self, data: dict) -> dict:
        """Transform data with processor.

        Parameters
        ----------
        data: dict
            data dictionary as gotten from TreeProcessor

        Returns
        -------
        dict:
            Data dictionary with normalized literal values.
        """
        # Assert that processor is fit.
        assert self._stats is not None, "Need to fit before transforming."
        # Get all data
        mappings, values = data['literal_mappings'], data['literal_values']
        # Make iterator
        iterator = zip(mappings, values)
        if self.verbose:
            tqdm(iterator)
        # Iterate over instances in data
        for idx, (mapping, values_) in enumerate(iterator):
            nodes = mapping[:, 0]
            types = mapping[:, 1]
            for node in torch.unique(nodes):
                mask = nodes == node
                key = '+'.join([str(t) for t in types[mask].tolist()])
                if '-1' in key:
                    continue
                values_[mask] = self.normalize(values_[mask], key)
            # Make nan's 0
            values_[values_.isnan()] = 0.
            # In place change of literal values to normalized ones
            data['literal_values'][idx] = values_
        return data
    
    def normalize(self, values, key):
        # Get statistics for type
        stats = self._stats.get(key, {'std': 1, 'mean': 0, 'min': 0, 'max': 2})
        # Normalize standard if std not zero
        if stats['std'] != 0:
            values = (values - stats['mean']) / (stats['std'] + 10e-4)
        # Normalize min max if std is zero
        else:
            values = 2 * (values - stats['min']) / (stats['max'] - stats['min'] + 10e-4) - 1
        return values

    def fit_transform(self, data: dict) -> dict:
        """Fit and transform processor.

        Parameters
        ----------
        data: dict
            Data to fit and transform

        Returns
        -------
        dict:
            Transformed data dictionary
        """
        self.fit(data)
        return self.transform(data)


import logging
from pandas import DataFrame, Series
import torch_geometric.data as Data


class DataProcessor():
    """Processing object for data files"""

    def __init__(self):
        """Initialize data processor.

        Parameters
        ----------
        files: Dict[str, DataFrame]
            Dictionary containing dataframes.
        """

    def apply_func_to_columns(self, file, columns, func, args):
        """Apply function to columns.

        Parameters
        ----------
        filename: str
            Name of file to access
        columns: List[str]
            Name of columns to apply function to
        func: callable
            function to apply
        args: Optional[Dict[str, Any]]
            Optional dictionary for function arguments
        """
        for col in columns:
            file[col] = file[col].apply(func) if args is None else file[col].apply(func, **args)
        return file

    def apply_func_to_df(self, file, func, **args):
        file = func(file, **args)
        return file

    def add_column_func(self, file, name, func: Optional[callable] = None, args: Optional[Dict] = None):
        """Add column to dataframe based on function or empty series.

        Parameters
        ----------
        filename: str
            name of file to acces
        Name: str
            Name of column to add
        func: Optional[callable]
            Function to apply.
            Defaults to none and empty column will be added.
        args: Optional[Dict[str, Any]]
            Optional dictionary for function arguments
        """
        if args is not None and func is None:
            raise ValueError('Arguments given, but no function')
        if func is not None:
            file[name] = file.apply(func, **args, axis=1) if args is not None else file.apply(func, axis=1)
        else:
            file[name] = Series([func] * file.index.size)
        return file

    def mask_df(self, file, mask):
        return mask_df(file, mask)

    def mask_on_cond_df(self, file: DataFrame, condition: callable, column: str):
        mask = condition(file[column])
        return mask_df(file, mask)

class GraphBuilder():
    """Class that builds KG graph from data"""

    def __init__(self, output_directory,
                 rel_ns: rdf.Namespace = rdf.namespace.Namespace("http://example.org/"),
                 node_ns: rdf.Namespace = rdf.namespace.Namespace("http://example.org/")):
        self.output_dir = output_directory
        self.rel_ns = rel_ns
        self.node_ns = node_ns
        # Make graph
        self.graph = rdf.Graph()

    def reset(self):
        self.graph = rdf.Graph()

    def load(self, name, format):
        self.graph = rdf.Graph()
        with open(join(self.output_dir, name + '.' + format), 'rb') as f:
            self.graph.parse(file=f, format=format)

    def save(self, name, format):
        self.graph.serialize(join(self.output_dir, name + "." + format), format=format)

    def remove_triple(self, triple: Tuple[Node, Node, Node]):
        self.graph.remove(triple)

    def remove_graph(self, graph,
                     condition: Tuple[Optional[Node], Optional[Node], Optional[Node]] = (None, None, None)):
        for triple in graph.triples(condition):
            self.remove_triple(triple)

    def add_triple(self, triple: Tuple[Node, Node, Node]):
        self.graph.add(triple)

    def add_graph(self, graph,
                  condition: Tuple[Optional[Node], Optional[Node], Optional[Node]] = (None, None, None)):
        for triple in graph.triples(condition):
            self.add_triple(triple)

    def add_from_df(self,
                    df: DataFrame,
                    triple_names: Iterable[Tuple[str, str, str]],
                    is_literal: bool = False
                    ):
        columns = df.columns
        n = df.index.size
        for src, rel, dest in triple_names:
            assert src in columns, f"{src} not in columns. "
            assert dest in columns, f"{dest} not in columns. "
            dest_ns = self.node_ns if not is_literal else None
            src_uri: List[Node] = df[src].apply(lambda s: self._enc(src, s, self.node_ns)).to_list()
            dest_uri: List[Node] = df[dest].apply(lambda s: self._enc(dest, s, dest_ns)).to_list()
            rel_uri: List[Node] = [self._enc(rel, None, self.rel_ns)]
            triples: Iterable[Tuple[Node, Node, Node]] = zip(src_uri, rel_uri * n, dest_uri)
            for triple in triples:
                self.graph.add(triple)

    @staticmethod
    def _enc(name: str, s: Optional[Any], ns: Optional[rdf.Namespace]) -> Node:
        name += '_' + str(s) if s is not None else ''
        return ns[name] if ns is not None else rdf.Literal(s)

class hide_stdout(object):
    def __enter__(self):
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args):
        logging.getLogger().setLevel(self.level)

class GraphProcessor():
    def __init__(self):
        super(GraphProcessor, self).__init__()
        # Namespaces
        self.namespaces = {
            'ex': rdf.namespace.Namespace("http://example.org/"),
            'sc': rdf.namespace.Namespace("http:schema.torch_geometric/"),
            'foaf': rdf.namespace.Namespace("http://xmlns.com/foaf/"),
            'rdf_ns': rdf.namespace.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
        }
        # Initialize labels list with none when processor is not yet built
        self.labels = None
        self.labels_dict = None
        self.nodes_dict = None

    @staticmethod
    def load_graph(graph, ext='nt'):
        if isinstance(graph, str):
            # Load graph
            with hide_stdout():
                g = rdf.Graph()
                with open(graph, 'rb') as f:
                    g.parse(file=f, format=ext)
            return g
        elif isinstance(graph, rdf.Graph):
            return graph
        else:
            raise ValueError('Graph must be RDF graph or path to graph file in .nt format')

    @staticmethod
    def get_stats(graph: rdf.graph) -> Tuple[List, set, set, List, List]:
        # Count predicates
        freq = Counter(graph.predicates())

        # Make relation, subject and object lists
        relations = sorted(set(graph.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(graph.subjects())
        objects = set(graph.objects())
        nodes = [node for node in subjects.union(objects) if not isinstance(node, rdf.Literal)]
        literals = [node for node in subjects.union(objects) if isinstance(node, rdf.Literal)]
        literal_types = list(set([lit.datatype for lit in literals]))
        return (relations, subjects, objects, nodes, literal_types)

    def build(self, graph: Union[str, rdf.Graph], labels):
        """Build internal state from training graph.

        Parameters
        ----------
        graph: rdf graph
            Graph from which to build state
        labels: List[Node]
            Lit of relationships to be considered as labels

        Returns
        -------

        """
        graph = self.load_graph(graph)
        # Set label object
        self.labels = labels

        # Get graph statistics
        relations, subjects, objects, nodes, literal_types = self.get_stats(graph)

        # Make relations, labels, nodes and literals dict
        self.relations_dict = {rel: i for i, rel in enumerate(relations) if rel not in self.labels}
        self.labels_dict = {label: i for i, label in enumerate(labels)}
        self.nodes_dict = {node: i for i, node in enumerate(nodes)}
        self.literal_types_dict = {lit: i for i, lit in enumerate(literal_types)}

    def process(self, graph: Union[str, rdf.Graph]
                ) -> Union[Tuple[Data.Data, dict], Tuple[Data.Data, dict, list, List[Union[int, Any]], List[int]]]:
        assert self.labels is not None, 'Processor not yet built. Build using a training graph and the `build´function.'
        graph = self.load_graph(graph)
        # Get graph stats
        relations, subjects, objects, nodes, literal_types = self.get_stats(graph)

        # Number of nodes
        num_nodes = len(nodes)
        num_relations = len([rel for rel in relations if rel not in self.labels])
        num_literals = 0

        # Make relations, labels, nodes and literals dict
        relations_dict = {rel: i for i, rel in enumerate(relations) if rel not in self.labels}
        nodes_dict = {node: i for i, node in enumerate(nodes)}
        literal_types_dict = {lit_type: num_nodes + i for i, lit_type in enumerate(literal_types)}

        # Initialize edges,  and node features
        edges = []
        # Initialize labels and corresponding train indices
        labels = {l: [] for l in self.labels}
        train_idx = {l: [] for l in self.labels}
        # Initialize node features and corresponding type
        literal_feature_values = []
        literal_feature_types = []

        # Initialize mappings
        node_mapping = [[], []]
        literal_type_mapping = [[], []]

        # Keep track of edge index where literal is present
        literal_edge_idx = []

        # Make mappings
        for literal in literal_types:
            # Destination index in built graph
            dst_ = self.literal_types_dict.get(literal, -1)
            # Destination index in current graph
            dst = literal_types_dict.get(literal)
            # Add map to mapping for nodes
            literal_type_mapping[0].append(dst)
            literal_type_mapping[1].append(dst_)
        for node in nodes:
            # Destination index in built graph
            dst_ = self.nodes_dict.get(node, -1)
            # Destination index in current graph
            dst = nodes_dict.get(node)
            # Add map to mapping for nodes
            node_mapping[0].append(dst)
            node_mapping[1].append(dst_)

        # Iterate triples
        for s, p, o in graph.triples((None, None, None)):
            # Flag whether object is literal
            is_literal = isinstance(o, rdf.Literal)
            is_blank = isinstance(o, rdf.BNode)

            # Source, destination and relation index in processed graph
            src = nodes_dict.get(s)
            rel = relations_dict.get(p)
            dst = nodes_dict.get(o)

            # If relation type is a label type: save label and corresponding node index
            # Skip to begin loop in order to remove edge ad node from graph
            if p in self.labels:
                # Add label
                labels[p].append(o.value)
                # Add train index
                train_idx[p].append(src)
                continue

            # Get literal values
            if is_literal:
                # Overwrite destination with literal type
                dst = literal_types_dict.get(o.datatype, -1)
                # Add literal value and type embedding
                literal_feature_types.append(dst)
                literal_feature_values.append(o.value)
                # Keep track of edge number
                literal_edge_idx.append(len(edges))
                literal_edge_idx.append(len(edges) + 1)
                # Increment literal
                num_literals += 1

            # Append triples to edges
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        # Process labels to integers
        label_dict = {p: {label: i for i, label in enumerate(set(labels_list))} for p, labels_list in
                      labels.items()}
        labels = {p: [label_dict[p][label_value] for label_value in labels_list] for p, labels_list in
                  labels.items()}

        # Create edge tensor and permute
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        perm = (num_nodes * num_relations * edges[0] + num_relations * edges[1] + edges[2]).argsort()
        edges = edges[:, perm]

        # Create tensor with all labels, indices and mapping to label type
        y = [torch.tensor(values, dtype=torch.long) for values in labels.values()]
        pos = [torch.tensor(idx, dtype=torch.long) for idx in train_idx.values()]
        y_map = [
            torch.tensor([self.labels_dict.get(label)] * len(values), dtype=torch.long)
            for label, values in labels.items()
        ]

        # build data object
        data = Data.Data(
            x=None,
            edge_index=edges[:2].t(),
            edge_type=edges[2],
            node_mapping=torch.tensor(node_mapping, dtype=torch.long).t(),
            literal_type_mapping=torch.tensor(literal_type_mapping, dtype=torch.long).t(),
            y=torch.cat(y, dim=0),
            pos=torch.cat(pos, dim=0),
            y_map=torch.cat(y_map, dim=0),
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_literals=num_literals,
        )

        if num_literals > 0:
            return (data, label_dict, literal_feature_values, literal_feature_types, literal_edge_idx)
        return (data, label_dict)

    # def process_tree(self, graph, root_node):
    #     max_depth = 2
    #     for level in range(max_depth):
    #         # Iterate triples
    #         for s, p, o in graph.triples((root_node, None, None)):
    #             # Flag whether object is literal
    #             is_literal = isinstance(o, rdf.Literal)
    #             is_blank = isinstance(o, rdf.BNode)
    #
    #             # Source, destination and relation index in processed graph
    #             src = nodes_dict.get(s)
    #             rel = relations_dict.get(p)
    #             dst = nodes_dict.get(o)
    #
    #             # If relation type is a label type: save label and corresponding node index
    #             # Skip to begin loop in order to remove edge ad node from graph
    #             if p in self.labels:
    #                 # Add label
    #                 labels[p].append(o.value)
    #                 # Add train index
    #                 train_idx[p].append(src)
    #                 continue
    #
    #             # Get literal values
    #             if is_literal:
    #                 # Overwrite destination with literal type
    #                 dst = literal_types_dict.get(o.datatype, -1)
    #                 # Add literal value and type embedding
    #                 literal_feature_types.append(dst)
    #                 literal_feature_values.append(o.value)
    #                 # Keep track of edge number
    #                 literal_edge_idx.append(len(edges))
    #                 literal_edge_idx.append(len(edges) + 1)
    #                 # Increment literal
    #                 num_literals += 1
    #
    #             # Append triples to edges
    #             edges.append([src, dst, 2 * rel])
    #             edges.append([dst, src, 2 * rel + 1])
        #     return tree

'''