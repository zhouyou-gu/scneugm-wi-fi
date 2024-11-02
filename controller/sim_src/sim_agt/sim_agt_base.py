import numpy as np
from scipy.sparse import csr_matrix
from sim_src.sim_env.env import WiFiNet

from sim_src.util import STATS_OBJECT

class sim_agt_base(STATS_OBJECT):
    TWT_START_TIME = 1e6
    TWT_ASLOT_TIME = 250
    def __init__(self):
        self.env:WiFiNet = None
        self.cfg = {}        
    
    def set_env(self,env):
        self.env = env
        
    def get_config(self):
        return {}

    def get_action(self):
        # heursitic method that assigns each user with a slot
        print("WARNING: using the base actions")
        return self.convert_action(np.arange(self.env.n_sta))
    
    def convert_action(self,act):
        twt_cfg = {}
        twt_cfg['twtstarttime'] = np.ones(act.size)*self.TWT_START_TIME
        twt_cfg['twtoffset'] = act*self.TWT_ASLOT_TIME
        twt_cfg['twtduration'] = np.ones(act.size)*self.TWT_ASLOT_TIME
        twt_cfg['twtperiodicity'] = np.ones(act.size)*self.TWT_ASLOT_TIME * (np.max(act)+1)
        return twt_cfg
    
    
    @staticmethod
    def greedy_coloring(adj_matrix, case='both'):
        """
        Perform graph coloring on a directed graph using a greedy algorithm.

        Parameters:
        adj_matrix: adjacency matrix of the graph.
        case (str): Specifies the case for coloring.
                    'incoming' - node has different color from in-coming neighbors.
                    'outgoing' - node has different color from out-going neighbors.
                    'both'     - node has different color from both in-coming and out-going neighbors.

        Returns:
        np.ndarray: An array where the index represents the vertex and the value represents its color.
        """
        n = adj_matrix.shape[0]  # Number of vertices
        colors = np.full(n, -1, dtype=int)  # Initialize all colors to -1

        adj_matrix =  csr_matrix(adj_matrix)
        adj_matrix.eliminate_zeros()
        # For in-coming neighbors, use CSC format for efficient column access
        adj_matrix_csc = adj_matrix.tocsc()

        for u in range(n):
            used_colors = set()

            if case in ['incoming', 'both']:
                # In-coming neighbors: nodes v such that there is an edge from v to u
                col_start = adj_matrix_csc.indptr[u]
                col_end = adj_matrix_csc.indptr[u + 1]
                in_neighbors = adj_matrix_csc.indices[col_start:col_end]
                used_colors.update(colors[in_neighbors])

            if case in ['outgoing', 'both']:
                # Out-going neighbors: nodes v such that there is an edge from u to v
                row_start = adj_matrix.indptr[u]
                row_end = adj_matrix.indptr[u + 1]
                out_neighbors = adj_matrix.indices[row_start:row_end]
                used_colors.update(colors[out_neighbors])

            # Remove the color -1 (unassigned colors)
            used_colors.discard(-1)

            # Assign the smallest available color
            color = 0
            while color in used_colors:
                color += 1
            colors[u] = color

        return colors
    
    
    @staticmethod
    def get_adj_matrix_from_edge_index(K, edge_index):
        assert edge_index.shape[0] == 2, "edge_index should be in (K,2)"
        data = np.ones(edge_index.shape[1], dtype=int)
        row_indices = edge_index[0, :]
        col_indices = edge_index[1, :]

        adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(K, K))
        
        adjacency_matrix += adjacency_matrix.T
        adjacency_matrix[adjacency_matrix>0] = 1
        adjacency_matrix.setdiag(0)
        
        return np.asarray(adjacency_matrix.todense())

    @staticmethod
    def get_same_color_edges(x, edges=None):
        """
        Returns the list of edge indices between nodes with the same color, including both directions.

        Parameters:
        x (array-like): Array where x[i] is the color index of node i.
        edges (array-like, optional): Array of edges in shape (2, K), where each column represents an edge (u, v).
                                    If not provided, assumes a fully connected graph.

        Returns:
        numpy.ndarray: Array of edges between nodes of the same color, including both directions.
        """
        num_nodes = x.shape[0]

        if edges is None:
            # Generate edges for a fully connected graph (excluding self-loops)
            u, v = np.triu_indices(num_nodes, k=1)
            edges = np.vstack((u, v))  # Shape (2, K)
        else:
            edges = np.asarray(edges)
            # Ensure edges are in shape (2, K)
            if edges.shape[0] != 2:
                if edges.shape[1] == 2:
                    edges = edges.T
                else:
                    raise ValueError("Edges should be in shape (2, K)")

        # Get colors of nodes at both ends of each edge
        x_u = x[edges[0]]
        x_v = x[edges[1]]

        # Find edges where both nodes have the same color
        same_color = x_u == x_v
        same_color_edges = edges[:, same_color]

        # Include both directions
        reverse_edges = same_color_edges[::-1, :]
        all_edges = np.hstack((same_color_edges, reverse_edges))
        
        # Remove duplicates
        all_edges_T = all_edges.T  # Shape (K, 2)
        unique_edges_T = np.unique(all_edges_T, axis=0)
        unique_edges = unique_edges_T.T
        
        return unique_edges
    
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
    
class agt_for_training(sim_agt_base):
    def __init__(self):
        super(agt_for_training, self).__init__()
        self.action = None
    
    def set_action(self,act):
        self.action = self.convert_action(act)
    
    def get_action(self):
        if self.action is None:
            raise Exception(self.__class__.__name__,"No action has been set")
        return self.action