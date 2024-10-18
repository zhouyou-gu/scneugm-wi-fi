import numpy as np
from scipy.sparse import csr_matrix
from sim_src.sim_env.env import WiFiNet

from sim_src.util import STATS_OBJECT
class sim_agt_base(STATS_OBJECT):
    TWT_START_TIME = 1e6
    TWT_ASLOT_TIME = 5e2
    def __init__(self):
        self.env:WiFiNet = None
        self.cfg = {}
    
    def set_env(self,env):
        self.env = env
        
    def get_config(self):
        return {}

    def get_action(self):
        # heursitic method that assigns each user with a slot
        return self.convert_action(np.arange(self.env.n_sta))
    
    def convert_action(self,act):
        twt_cfg = {}
        twt_cfg['twtstarttime'] = np.ones(act.size)*self.TWT_START_TIME
        twt_cfg['twtoffset'] = act*self.TWT_ASLOT_TIME
        twt_cfg['twtduration'] = np.ones(act.size)*self.TWT_ASLOT_TIME
        twt_cfg['twtperiodicity'] = np.ones(act.size)*self.TWT_ASLOT_TIME * (np.max(act)+1)
        return twt_cfg
    
    
    @staticmethod
    def greedy_coloring(adj_matrix:csr_matrix, case='both'):
        """
        Perform graph coloring on a directed graph using a greedy algorithm.

        Parameters:
        adj_matrix (csr_matrix): Sparse adjacency matrix of the graph in CSR format.
        case (str): Specifies the case for coloring.
                    'incoming' - node has different color from in-coming neighbors.
                    'outgoing' - node has different color from out-going neighbors.
                    'both'     - node has different color from both in-coming and out-going neighbors.

        Returns:
        np.ndarray: An array where the index represents the vertex and the value represents its color.
        """
        n = adj_matrix.shape[0]  # Number of vertices
        colors = np.full(n, -1, dtype=int)  # Initialize all colors to -1

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