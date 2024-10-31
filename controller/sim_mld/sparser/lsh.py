import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix, triu

import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix, triu

class LSH:
    def __init__(self, num_bits, num_tables, bits_per_hash):
        self.num_bits = num_bits              # Length of binary vectors
        self.num_tables = num_tables          # Number of hash tables (T)
        self.bits_per_hash = bits_per_hash    # Number of bits per hash function (k)
        self.hash_tables = [dict() for _ in range(num_tables)]
        self.hash_functions = [
            np.random.choice(num_bits, bits_per_hash, replace=False)
            for _ in range(num_tables)
        ]
        self.vectors = None  # To store the binary vectors

    def build_hash_tables(self, vectors):
        """
        Build hash tables from a list of binary vectors in batch without explicit loops over vectors.
        
        Parameters:
            vectors (list or numpy.ndarray): List or array of binary vectors.
        """
        vectors = np.asarray(vectors, dtype=np.uint8)  # Ensure binary data is of integer type
        num_vectors = vectors.shape[0]
        vector_ids = np.arange(num_vectors)
        self.num_vectors = num_vectors  # Store for later use
        self.vectors = vectors  # Store vectors for Hamming distance computations

        for t in range(self.num_tables):
            indices = self.hash_functions[t]
            # Extract bits at selected indices for all vectors
            hash_values = vectors[:, indices]
            # Convert binary hash values to integer hash codes
            hash_codes = self._hash_codes(hash_values)
            # Group vector IDs by hash codes without explicit loops
            hash_table = self._group_vector_ids(hash_codes, vector_ids)
            self.hash_tables[t] = hash_table

    def _hash_codes(self, hash_values):
        """
        Convert binary hash values to integer hash codes.
        
        Parameters:
            hash_values (numpy.ndarray): Binary hash values of shape (num_vectors, bits_per_hash).
        
        Returns:
            numpy.ndarray: Array of integer hash codes.
        """
        # Convert binary vectors to integer hash codes
        powers_of_two = 1 << np.arange(self.bits_per_hash)[::-1]
        hash_codes = np.dot(hash_values, powers_of_two)
        return hash_codes

    def _group_vector_ids(self, hash_codes, vector_ids):
        """
        Group vector IDs by hash codes using NumPy operations.
        
        Parameters:
            hash_codes (numpy.ndarray): Array of integer hash codes.
            vector_ids (numpy.ndarray): Array of vector IDs.
        
        Returns:
            dict: Dictionary mapping hash codes to lists of vector IDs.
        """
        # Sort hash codes and corresponding vector IDs
        sorted_indices = np.argsort(hash_codes)
        sorted_hash_codes = hash_codes[sorted_indices]
        sorted_vector_ids = vector_ids[sorted_indices]

        # Find unique hash codes and their starting indices
        unique_hash_codes, start_indices, counts = np.unique(
            sorted_hash_codes, return_counts=True, return_index=True
        )

        # Split the sorted vector IDs into groups based on hash codes
        split_indices = np.cumsum(counts[:-1])
        grouped_vector_ids = np.split(sorted_vector_ids, split_indices)

        # Build the hash table as a dictionary without explicit loops
        hash_table = dict(zip(unique_hash_codes, grouped_vector_ids))

        return hash_table

    def export_adjacency_matrix(self, hamming_threshold=None):
        """
        Export the colliding edges as a binary adjacency matrix.
        Optionally filters pairs based on Hamming distance.

        Parameters:
            hamming_threshold (int, optional): Maximum Hamming distance to include a pair. 
                                               If None, all colliding pairs are included.

        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix of size (N, N).
        """
        if self.vectors is None:
            raise ValueError("Vectors have not been stored. Please call build_hash_tables first.")

        num_vectors = self.num_vectors
        rows = []
        cols = []

        vectors = self.vectors  # For easier access

        for t in range(self.num_tables):
            hash_table = self.hash_tables[t]
            for vector_ids in hash_table.values():
                if len(vector_ids) > 1:
                    # Create all unique pairs of vector IDs within the bucket
                    vector_ids = np.array(vector_ids)
                    idx1, idx2 = np.triu_indices(len(vector_ids), k=1)
                    pair1 = vector_ids[idx1]
                    pair2 = vector_ids[idx2]

                    if hamming_threshold is not None:
                        # Compute Hamming distances for all pairs
                        # XOR the vectors and sum the differing bits
                        xor = vectors[pair1] != vectors[pair2]
                        hamming_distances = np.sum(xor, axis=1)
                        # Filter pairs where Hamming distance is less than the threshold
                        valid = hamming_distances < hamming_threshold
                        pair1 = pair1[valid]
                        pair2 = pair2[valid]

                    # Append to rows and cols
                    rows.extend(pair1)
                    cols.extend(pair2)

        if not rows:
            # No edges to add
            adjacency_matrix = csr_matrix((num_vectors, num_vectors), dtype=np.uint8)
            return adjacency_matrix

        # Create symmetric adjacency matrix
        data = np.ones(len(rows), dtype=np.uint8)
        adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(num_vectors, num_vectors))
        # Since the adjacency matrix is symmetric, add the transpose
        adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose()
        # Remove any duplicate entries
        adjacency_matrix.data = np.clip(adjacency_matrix.data, 0, 1)
        # Convert to CSR format for efficient arithmetic and matrix vector operations
        adjacency_matrix = adjacency_matrix.tocsr()
        return adjacency_matrix

    @staticmethod
    def compare_adjacency_matrices(exported_matrix, target_matrix):
        """
        Compare two adjacency matrices and compute false positives and false negatives.

        Parameters:
            exported_matrix (scipy.sparse.csr_matrix): The adjacency matrix exported from LSH.
            target_matrix (scipy.sparse.csr_matrix): The target adjacency matrix.

        Returns:
            dict: A dictionary containing counts of TP, FP, FN, TN, and rates.
        """
        # Ensure matrices are in CSR format for efficient arithmetic operations
        exported_matrix = exported_matrix.tocsr()
        target_matrix = target_matrix.tocsr()

        # Number of possible edges (excluding self-loops)
        num_vectors = exported_matrix.shape[0]
        num_possible_edges = num_vectors * (num_vectors - 1) // 2

        # Extract the upper triangle (since the adjacency matrices are symmetric)
        exported_upper = triu(exported_matrix, k=1)
        target_upper = triu(target_matrix, k=1)

        # Convert to boolean format (non-zero entries are True)
        exported_edges = exported_upper.astype(bool)
        target_edges = target_upper.astype(bool)

        P = target_edges.nnz
        N = num_possible_edges - target_edges.nnz

        AP = exported_edges.nnz
        AN = num_possible_edges - exported_edges.nnz

        # Compute True Positives (TP): edges present in both matrices
        tp_matrix = exported_edges.multiply(target_edges)
        TP = tp_matrix.nnz

        # Compute False Positives (FP): edges in exported but not in target
        fp_matrix = exported_edges - tp_matrix
        FP = fp_matrix.nnz

        # Compute False Negatives (FN): edges in target but not in exported
        fn_matrix = target_edges - tp_matrix
        FN = fn_matrix.nnz

        # Compute True Negatives (TN): non-edges in both matrices
        TN = num_possible_edges - TP - FP - FN

        # Compute rates
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Return results
        results = {
            'Total Edges': num_possible_edges,
            'Target Positives': P,
            'Target Negatives': N,
            'Approx Positives': AP,
            'Approx Negatives': AN,
            'True Positives': TP,
            'False Positives': FP,
            'False Negatives': FN,
            'True Negatives': TN,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }
        return results
    
    @staticmethod
    def query_rows(binary_matrix, n=3, max_attempts=1000, target_matching=20):
        """
        Repeatedly queries the binary matrix by selecting random column positions and binary vectors
        until a target number of unique matching rows is found. If an attempt yields more matches
        than needed, a subset is randomly selected to meet the target.

        Parameters:
        - binary_matrix (np.ndarray): The binary matrix of shape (K, M).
        - n (int): The number of column positions to select in each attempt.
        - max_attempts (int): Maximum number of attempts to find matching rows.
        - target_matching (int): The exact number of unique matching rows desired.

        Returns:
        - matching_row_indices (np.ndarray): Array of matching row indices with length `target_matching`.
        - match_mask (np.ndarray): Boolean array of shape (K,) where True indicates a matching row.

        Raises:
        - ValueError: If the target number of matching rows cannot be found within `max_attempts`.
        """
        K, M = binary_matrix.shape

        # Input validations
        if n > M:
            raise ValueError("n cannot be greater than the number of columns M.")
        if target_matching < 1:
            raise ValueError("target_matching must be at least 1.")
        if target_matching > K:
            raise ValueError("target_matching cannot exceed the number of rows K in the matrix.")

        accumulated_matching_rows = set()
        attempts = 0

        while attempts < max_attempts and len(accumulated_matching_rows) < target_matching:
            attempts += 1
            # Step 1: Randomly select n unique column indices
            selected_positions = np.random.choice(M, size=n, replace=False)

            # Step 2: Generate a random binary vector of length n
            binary_vector = np.random.randint(0, 2, size=n)

            # Step 3: Extract the submatrix corresponding to the selected positions
            submatrix = binary_matrix[:, selected_positions]

            # Step 4: Create a boolean mask where rows match the binary vector
            match_mask = np.all(submatrix == binary_vector, axis=1)

            # Step 5: Get the indices of rows where match_mask is True
            matching_row_indices = np.where(match_mask)[0]

            # Step 6: Identify new matching rows not yet accumulated
            new_matching_rows = [row for row in matching_row_indices if row not in accumulated_matching_rows]

            # If new matching rows are found, add them to the accumulated set
            if new_matching_rows:
                for row in new_matching_rows:
                    if len(accumulated_matching_rows) < target_matching:
                        accumulated_matching_rows.add(row)
                    if len(accumulated_matching_rows) == target_matching:
                        break  # Target achieved

        # After all attempts, check if the target was achieved
        if len(accumulated_matching_rows) < target_matching:
            raise ValueError(f"Could not find {target_matching} matching rows within {max_attempts} attempts.")

        # Convert the accumulated set to a sorted list for consistency
        accumulated_matching_rows = sorted(accumulated_matching_rows)

        # If more rows are accumulated than needed (due to simultaneous additions), randomly select the required number
        if len(accumulated_matching_rows) > target_matching:
            selected_indices = np.random.choice(len(accumulated_matching_rows), size=target_matching, replace=False)
            final_matching_rows = np.array([accumulated_matching_rows[i] for i in selected_indices])
        else:
            final_matching_rows = np.array(accumulated_matching_rows)

        # Create the match mask with exactly target_matching True values
        match_mask = np.zeros(K, dtype=bool)
        match_mask[final_matching_rows] = True

        return final_matching_rows, match_mask



if __name__ == "__main__":
    import numpy as np
    from working_dir_path import get_controller_path
    from sim_mld.sparse_transformer.sparser.model import sparser_base
    from sim_mld.sparse_transformer.tokenizer.model import tokenizer_base
    from sim_src.sim_env.env import WiFiNet

    from sim_src.util import *

    np.set_printoptions(precision=3)

    # load tokenizer model
    tk_model = tokenizer_base()
    path = get_controller_path()
    path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/tokenizer_base.final.pt")
    tk_model.load_model(path=path)
    tk_model.eval()

    # load sparser model
    sp_model = sparser_base()
    path = get_controller_path()
    path = os.path.join(path, "sim_alg/train_sparser/selected_nn/sparser_base.final.pt")
    sp_model.load_model(path=path)
    sp_model.eval()


    N_TRAINING_STEP = 10
    for i in range(N_TRAINING_STEP):
        # get network state 
        e = WiFiNet(seed=i)
        b = e.get_sta_states()
        
        # tokenize sta states
        l, _ = tk_model.get_output_np_batch(b)
        
        # get collision matrix
        target_collision_matrix = e.get_CH_matrix()
        target_collision_matrix = csr_matrix(target_collision_matrix).astype(np.int8)
        target_collision_matrix.eliminate_zeros()
        
        # get hard code
        hc = sp_model.get_output_np(l)
        hc = sp_model.binarize_hard_code(hc)
        
        # lsh
        lsh = LSH(num_bits=sp_model.model.hash_dim, num_tables=30, bits_per_hash=6)
        lsh.build_hash_tables(hc)
        approx_collision_matrix = lsh.export_adjacency_matrix()
        res = lsh.compare_adjacency_matrices(approx_collision_matrix,target_collision_matrix)
        print(e.n_sta**2,res)


        I = e.get_interfering_node_matrix()
        I = csr_matrix(I).astype(np.int8)
        I.eliminate_zeros()
        res = lsh.compare_adjacency_matrices(I,target_collision_matrix)
        print(e.n_sta**2,res)
