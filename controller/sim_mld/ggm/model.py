import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.ggm.nn import GraphGenerator
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch import optim
import networkx as nx

class GGM(base_model):
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR, WITH_TARGET = False)
    
    def init_model(self):
        self.model = GraphGenerator()
    
    def init_optim(self):
        self.eva_optim = optim.Adam(self.model.graph_evaluator.parameters(), lr=self.LR)
        self.gen_optim = optim.Adam(self.model.graph_generator.parameters(), lr=self.LR)
        
    @counted
    def step(self, batch):        
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        
        x = to_tensor(batch["x"])
        token = to_tensor(batch["token"])
        edge_value_action = to_tensor(batch["edge_value"])
        edge_attr = to_tensor(batch["edge_attr"])
        color_collision = to_tensor(batch["color_collision"])
        edge_attr_with_color_collision = torch.cat([edge_attr.unsqueeze(-1),color_collision.unsqueeze(-1)],dim=-1)
        edge_index = to_tensor(batch["edge_index"],dtype=LONG_INTEGER)
        q_target = to_tensor(batch["q"])
        nc = to_tensor(batch["nc"])
        n_sta = to_tensor(batch["n_sta"])
       
        q_approx_action = self.model.evaluate_graph(x,token,edge_value_action,edge_attr_with_color_collision,edge_index).squeeze()
        sum_edge_value_action = torch.zeros_like(q_approx_action).scatter_add_(0, edge_index[1], edge_value_action)
        loss_eva = nn.functional.binary_cross_entropy(q_approx_action, q_target, reduction="mean")
        self.eva_optim.zero_grad()
        loss_eva.backward()
        self.eva_optim.step()
        self.eva_optim.zero_grad()


        edge_value = self.model.generate_graph(x,token,edge_attr,edge_index).squeeze()
        q_approx = self.model.evaluate_graph(x,token,edge_value,edge_attr_with_color_collision,edge_index).squeeze()
        sum_edge_value = torch.zeros_like(q_approx).scatter_add_(0, edge_index[1], edge_value)
        # loss_gen = nn.functional.mse_loss(edge_value,(edge_attr>0).float(), reduction="mean")

        loss_gen = -q_approx.mean() 
 
        self.gen_optim.zero_grad()
        loss_gen.backward()
        self.gen_optim.step()
        self.gen_optim.zero_grad()
        
        s = ""
        s += f"K:{q_target.sum():>4.0f}/{n_sta.item():>4.0f}:{nc.item():>3.0f}"
        s += f", E:{edge_value_action.numel():>6d}"
        s += f", e+:{edge_value_action[edge_value_action>0].sum():>6.0f}"
        s += f", i+:{(edge_attr>0).sum():>6.0f}"
        s += f", s_e|i+:{edge_value_action[edge_attr>0].sum():>6.0f}"
        s += f", m_e:{edge_value_action.mean():>6.2f}"
        s += f", m_d|q+:{sum_edge_value_action[q_target>0].mean():>6.2f}"
        s += f", m_d|q0:{sum_edge_value_action[q_target==0].mean():>6.2f}"
        s += f", m_d/K|q+:{sum_edge_value_action[q_target>0].mean()/n_sta.item():>6.2f}"
        s += f", m_d/K|q0:{sum_edge_value_action[q_target==0].mean()/n_sta.item():>6.2f}"
        # s += f", c(e,i):{torch.corrcoef(torch.stack([edge_value, edge_attr>0]))[0, 1].item():>4.2f}"
        # s += f", loss_gen:{loss_gen.item():>4.2f}"
        self._printalltime(s)

        # self._printalltime(f"loss_eva: {loss_eva.item():.4f}, loss_gen: {loss_gen.item():.4f}")
        self._add_np_log("loss",self.N_STEP,loss_eva.item(),loss_gen.item())

    @torch.no_grad()
    def get_output_np_edge_weight(self, x, token, edge_attr, edge_index, exploration_p = 0.):
        x = to_tensor(x)
        token = to_tensor(token)
        edge_attr = to_tensor(edge_attr)
        edge_index = to_tensor(edge_index,dtype=LONG_INTEGER)
        
        edge_value = self.model.generate_graph(x,token,edge_attr,edge_index)        
        edge_value = to_numpy(edge_value).squeeze()
        if p_true(exploration_p):
            print("+++++ exploration +++++")
            edge_value = GGM.binarize_vector(edge_value)
        else:
            edge_value = edge_value > 0.5
            edge_value = edge_value.astype(float)
        return  edge_value
    
    @staticmethod
    def binarize_vector(prob_vector, seed=None):
        """
        Converts a vector of probabilities into a binary vector.
        Each element in the binary vector is 1 with probability equal to the corresponding element in prob_vector,
        and 0 otherwise.

        Parameters:
        - prob_vector (np.ndarray): 1D array with values in the range [0, 1], representing probabilities.
        - seed (int, optional): Seed for the random number generator for reproducibility.

        Returns:
        - binary_vector (np.ndarray): 1D binary array where each element is 0 or 1.
        """
        if not isinstance(prob_vector, np.ndarray):
            raise TypeError("prob_vector must be a NumPy array.")
        
        if prob_vector.ndim != 1:
            raise ValueError("prob_vector must be a 1D array.")
        
        if np.any(prob_vector < 0) or np.any(prob_vector > 1):
            raise ValueError("All elements in prob_vector must be in the range [0, 1].")
        
        rng = np.random.default_rng(seed)
        binary_vector = rng.binomial(1, prob_vector)
        return binary_vector
    
    @staticmethod
    def construct_adjacency_matrix(edge_index, edge_values, num_nodes=None, directed=True, include_self_loops=False):
        """
        Constructs an adjacency matrix from edge indices and edge values.

        Parameters:
        - edge_index (np.ndarray or list of lists): 2 x E array/list where E is the number of edges.
                                                The first row/list contains source node indices,
                                                and the second row/list contains target node indices.
        - edge_values (np.ndarray or list): E-dimensional array/list of edge values.
        - num_nodes (int, optional): Number of nodes in the graph. If not provided, it will be inferred
                                    from the maximum node index in edge_index.
        - directed (bool): Whether the graph is directed. Defaults to False (undirected).
        - include_self_loops (bool): Whether to include self-loops in the adjacency matrix. Defaults to False.

        Returns:
        - adjacency_matrix (np.ndarray): Square adjacency matrix of shape (num_nodes, num_nodes).
        """
        # Convert edge_index and edge_values to NumPy arrays if they aren't already
        edge_index = np.array(edge_index)
        edge_values = np.array(edge_values)
        
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a 2 x E array.")
        
        if edge_index.shape[1] != edge_values.shape[0]:
            raise ValueError("Number of edges in edge_index and edge_values must match.")
        
        # Infer number of nodes if not provided
        if num_nodes is None:
            num_nodes = max(edge_index.max(), edge_index.min() if edge_index.size > 0 else 0) + 1
        
        # Initialize adjacency matrix with zeros
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=edge_values.dtype)
        
        # Iterate over edges and assign values
        for src, tgt, val in zip(edge_index[0], edge_index[1], edge_values):
            if not include_self_loops and src == tgt:
                continue  # Skip self-loops if not included
            adjacency_matrix[src, tgt] = val
            if not directed:
                adjacency_matrix[tgt, src] = val  # Ensure symmetry for undirected graphs
        
        return adjacency_matrix
    
    @staticmethod
    def export_all_edges(adjacency_matrix, directed=True, include_self_loops=False):
        """
        Extracts all possible edge indices and their corresponding values from a given adjacency matrix,
        excluding self-loops.

        Parameters:
        - adjacency_matrix (np.ndarray): Square matrix representing the adjacency matrix of the graph.
        - directed (bool): Whether the graph is directed. Defaults to False (undirected).
        - include_self_loops (bool): Whether to include self-loops. Defaults to False.

        Returns:
        - edge_index (np.ndarray): 2 x E array where E is the number of edges.
        - edge_features (np.ndarray): E-dimensional array of edge values.
        """
        if not isinstance(adjacency_matrix, np.ndarray):
            raise TypeError("Adjacency matrix must be a NumPy array.")
        
        if adjacency_matrix.ndim != 2:
            raise ValueError("Adjacency matrix must be a 2D array.")
        
        num_rows, num_cols = adjacency_matrix.shape
        if num_rows != num_cols:
            raise ValueError("Adjacency matrix must be square.")
        
        num_nodes = num_rows
        
        if directed:
            # Generate all possible (i, j) pairs excluding self-loops
            src, tgt = np.where(~np.eye(num_nodes, dtype=bool))
        else:
            # Generate all possible (i, j) pairs in upper triangle excluding self-loops
            src, tgt = np.triu_indices(num_nodes, k=1)
        
        if include_self_loops:
            if directed:
                # Include all self-loops
                self_src, self_tgt = np.where(np.eye(num_nodes, dtype=bool))
                src = np.concatenate((src, self_src))
                tgt = np.concatenate((tgt, self_tgt))
            else:
                # Include self-loops in undirected graphs
                self_src, self_tgt = np.where(np.eye(num_nodes, dtype=bool))
                src = np.concatenate((src, self_src))
                tgt = np.concatenate((tgt, self_tgt))
        
        edge_index = np.vstack((src, tgt))
        edge_features = adjacency_matrix[src, tgt]
        
        return edge_features, edge_index

    @staticmethod
    def maximum_independent_set(adjacency_matrix):
        """
        Computes an approximate Maximum Independent Set (MIS) of a graph given its adjacency matrix
        using a greedy heuristic.
        
        Parameters:
        - adjacency_matrix (np.ndarray): A binary square matrix representing the adjacency matrix of the graph.
        
        Returns:
        - mis_mask (np.ndarray): A binary mask where elements corresponding to the MIS nodes are 1, and others are 0.
        """
        # Validate input types
        if not isinstance(adjacency_matrix, np.ndarray):
            raise TypeError("Adjacency matrix must be a NumPy array.")
        
        if adjacency_matrix.ndim != 2:
            raise ValueError("Adjacency matrix must be a 2D array.")
        
        num_rows, num_cols = adjacency_matrix.shape
        if num_rows != num_cols:
            raise ValueError("Adjacency matrix must be square.")
        
        # Create a NetworkX graph from the adjacency matrix
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Initialize the Maximum Independent Set
        mis = set()
        
        # Create a copy of the graph to manipulate
        G_copy = G.copy()
        
        # Greedy heuristic: iteratively select the node with the smallest degree
        while G_copy.number_of_nodes() > 0:
            # Select node with the smallest degree
            node = min(G_copy.nodes(), key=lambda n: G_copy.degree(n))
            mis.add(node)
            
            # Remove the node and its neighbors from the graph
            neighbors = list(G_copy.neighbors(node))
            G_copy.remove_node(node)
            G_copy.remove_nodes_from(neighbors)
        
        # Create a binary mask
        mis_mask = np.zeros(num_rows, dtype=int)
        mis_mask[list(mis)] = 1
        
        return mis_mask