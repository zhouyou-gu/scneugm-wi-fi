import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix, triu

class LSH:
    def __init__(self, num_bits, num_tables, bits_per_hash):
        self.num_bits = num_bits              # Length of binary vectors
        self.num_tables = num_tables          # Number of hash tables (T)
        self.bits_per_hash = bits_per_hash    # Number of bits per hash function (k)
        self.vectors = None  # To store the binary vectors

    def build_hash_tables(self, vectors):
        """
        Build hash tables from a list of binary vectors in batch without explicit loops over vectors.
        
        Parameters:
            vectors (list or numpy.ndarray): List or array of binary vectors.
        """
        self.hash_tables = [dict() for _ in range(self.num_tables)]
        self.hash_functions = [
            np.random.choice(self.num_bits, self.bits_per_hash, replace=False)
            for _ in range(self.num_tables)
        ]
        
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

    def insert_new_hash_table(self, vectors):
        """
        Insert a new hash table and remove the oldest hash table to maintain the number of hash tables.
        
        Raises:
            ValueError: If vectors have not been built yet.
        """
        vectors = np.asarray(vectors, dtype=np.uint8)  # Ensure binary data is of integer type
        num_vectors = vectors.shape[0]
        assert num_vectors == self.num_vectors, "currently, the number of vectors cannot be changed"
        self.num_vectors = num_vectors  # Store for later use
        self.vectors = vectors  # Store vectors for Hamming distance computations

        # Step 1: Generate a new hash function
        new_hash_function = np.random.choice(self.num_bits, self.bits_per_hash, replace=False)
        
        # Step 2: Hash all vectors using the new hash function
        hash_values = self.vectors[:, new_hash_function]
        hash_codes = self._hash_codes(hash_values)
        vector_ids = np.arange(self.num_vectors)
        new_hash_table = self._group_vector_ids(hash_codes, vector_ids)
        
        # Step 3: Append the new hash function and hash table
        self.hash_functions.append(new_hash_function)
        self.hash_tables.append(new_hash_table)
        
        # Step 4: Remove the oldest hash function and hash table to maintain the count
        self.hash_functions.pop(0)
        self.hash_tables.pop(0)

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

    def export_adjacency_matrix_with_mask_direct(self, mask, num_random_bits=4, hamming_threshold=None):
        """
        Export the adjacency matrix based on randomly selected bits from masked vectors.
        Pairs are formed between vectors that match on the selected random bits of any masked vector.

        Parameters:
            mask (numpy.ndarray or list of bool): Boolean mask array indicating which vectors are masked.
            num_random_bits (int): Number of random bits to select from each masked vector for querying.
            hamming_threshold (int, optional): Maximum Hamming distance to include a pair. 
                                               If None, all matching pairs are included.

        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix of size (N, N).
        """
        if self.vectors is None:
            raise ValueError("Vectors have not been stored. Please call build_hash_tables first.")

        # Convert mask to numpy array if it's a list
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.num_vectors:
            raise ValueError(f"Mask length {mask.shape[0]} does not match number of vectors {self.num_vectors}.")

        masked_vector_ids = np.nonzero(mask)[0]
        if masked_vector_ids.size == 0:
            return csr_matrix((self.num_vectors,self.num_vectors))
        
        num_vectors = self.num_vectors
        rows = []
        cols = []

        vectors = self.vectors  # For easier access
        num_bits = self.num_bits

        for vector_id in masked_vector_ids:
            # Randomly select bit positions for the current masked vector
            selected_bits = np.random.choice(num_bits, size=num_random_bits, replace=False)
            query_bits = vectors[vector_id, selected_bits]

            # Find all vectors that match the query bits
            matches = np.all(vectors[:, selected_bits] == query_bits, axis=1)
            matching_ids = np.nonzero(matches)[0]

            if matching_ids.size < 2:
                continue  # No pairs to form

            # Exclude the masked vector itself from matching_ids
            matching_ids = matching_ids[matching_ids != vector_id]

            if matching_ids.size == 0:
                continue  # No other vectors match

            # If hamming_threshold is specified, filter based on Hamming distance
            if hamming_threshold is not None:
                xor = vectors[vector_id] != vectors[matching_ids]
                hamming_distances = np.sum(xor, axis=1)
                valid = hamming_distances < hamming_threshold
                matching_ids = matching_ids[valid]

            if matching_ids.size == 0:
                continue  # No valid matches after filtering

            # Create pairs between the masked vector and matching vectors
            pair1 = np.full(matching_ids.shape, vector_id, dtype=int)
            pair2 = matching_ids

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
    def query_rows(binary_matrix, n=4, max_attempts=1000, target_matching=20):
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
    
    
    @staticmethod
    def export_all_edges_of_sparse_matrix(adj_matrix):
        """
        Export all edges from the adjacency matrix in both directions as a (2, E) NumPy array.

        Parameters:
            adj_matrix (scipy.sparse.csr_matrix): The adjacency matrix of size (N, N).

        Returns:
            numpy.ndarray: A (2, E) array where each column represents a directed edge (from, to).
        """
        if not isinstance(adj_matrix, csr_matrix):
            adj_matrix = adj_matrix.tocsr()
        
        # Extract the row and column indices of non-zero entries
        rows, cols = adj_matrix.nonzero()
        
        # Stack the row and column indices to form a (2, E) array
        edges = np.vstack((rows, cols))
        
        return edges