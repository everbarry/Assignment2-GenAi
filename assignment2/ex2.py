from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
import time


def load_dataset(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        dataset = np.array(list(reader)).astype(np.float32)
    return dataset


def print_tree(root, parents, children, node_names=None, indent="", is_last=True):
    """
    Print a tree structure in a graphical way in the CLI.

    Args:
        root: The root node index
        parents: List of parent indices for each node
        children: List of children lists for each node
        node_names: Optional list of node names (default: node indices)
        indent: Current indentation (used in recursion)
        is_last: Whether this node is the last child of its parent
    """
    if node_names is None:
        node_names = [str(i) for i in range(len(parents))]
    prefix = "└── " if is_last else "├── "
    print(f"{indent}{prefix}Node {node_names[root]}")
    child_indent = indent + ("    " if is_last else "│   ")
    for i, child in enumerate(children[root]):
        print_tree(child, parents, children, node_names, child_indent, i == len(children[root]) - 1)


class BinaryCLT:
    def __init__(self, data, root=None, alpha: float = 0.01):
        """
        Initialize and learn a Chow-Liu Tree from binary data.

        Args:
            data: Binary data matrix (samples x variables) with values in {0, 1}
            root: Root node index (or None for random root)
            alpha: Smoothing parameter for Laplace correction
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.alpha = alpha
        if root is None:
            self.root = np.random.randint(0, self.n_vars)
        else:
            assert 0 <= root < self.n_vars
            self.root = root
        self._learn_structure()
        self._compute_parameters()

    def _learn_structure(self):
        """Learn the tree structure using the Chow-Liu algorithm."""
        mi_matrix = np.zeros((self.n_vars, self.n_vars))
        marginal_counts = np.zeros((self.n_vars, 2))
        marginal_counts[:, 0] = np.sum(self.data == 0, axis=0) + self.alpha
        marginal_counts[:, 1] = np.sum(self.data == 1, axis=0) + self.alpha
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                counts_00 = np.sum((self.data[:, i] == 0) & (self.data[:, j] == 0)) + self.alpha
                counts_01 = np.sum((self.data[:, i] == 0) & (self.data[:, j] == 1)) + self.alpha
                counts_10 = np.sum((self.data[:, i] == 1) & (self.data[:, j] == 0)) + self.alpha
                counts_11 = np.sum((self.data[:, i] == 1) & (self.data[:, j] == 1)) + self.alpha
                p_00 = counts_00 / (self.n_samples + 4 * self.alpha)
                p_01 = counts_01 / (self.n_samples + 4 * self.alpha)
                p_10 = counts_10 / (self.n_samples + 4 * self.alpha)
                p_11 = counts_11 / (self.n_samples + 4 * self.alpha)
                p_i0 = marginal_counts[i, 0] / (2 * self.alpha + self.n_samples)
                p_i1 = marginal_counts[i, 1] / (2 * self.alpha + self.n_samples)
                p_j0 = marginal_counts[j, 0] / (2 * self.alpha + self.n_samples)
                p_j1 = marginal_counts[j, 1] / (2 * self.alpha + self.n_samples)
                mi = 0
                if p_00 > 0: mi += p_00 * np.log(p_00 / (p_i0 * p_j0))
                if p_01 > 0: mi += p_01 * np.log(p_01 / (p_i0 * p_j1))
                if p_10 > 0: mi += p_10 * np.log(p_10 / (p_i1 * p_j0))
                if p_11 > 0: mi += p_11 * np.log(p_11 / (p_i1 * p_j1))
                mi_matrix[i, j] = mi_matrix[j, i] = mi
        mst = minimum_spanning_tree(-mi_matrix).toarray()
        self.parents = np.full(self.n_vars, -1, dtype=int)
        self.children = [[] for _ in range(self.n_vars)]
        _, predecessors = breadth_first_order(mst, self.root, directed=False, return_predecessors=True)
        for i in range(self.n_vars):
            if i != self.root:
                parent = predecessors[i]
                self.parents[i] = parent
                self.children[parent].append(i)

    def _compute_parameters(self):
        self.log_theta = np.zeros((self.n_vars, 2, 2))
        root_counts = np.zeros(2)
        root_counts[0] = np.sum(self.data[:, self.root] == 0) + self.alpha
        root_counts[1] = np.sum(self.data[:, self.root] == 1) + self.alpha
        root_probs = root_counts / (2 * self.alpha + self.n_samples)
        self.log_root_probs = np.log(root_probs)
        for i in range(self.n_vars):
            if i == self.root:
                continue
            parent = self.parents[i]
            counts = np.zeros((2, 2))
            counts[0, 0] = np.sum((self.data[:, parent] == 0) & (self.data[:, i] == 0)) + self.alpha
            counts[0, 1] = np.sum((self.data[:, parent] == 0) & (self.data[:, i] == 1)) + self.alpha
            counts[1, 0] = np.sum((self.data[:, parent] == 1) & (self.data[:, i] == 0)) + self.alpha
            counts[1, 1] = np.sum((self.data[:, parent] == 1) & (self.data[:, i] == 1)) + self.alpha
            joint_probs = counts / (4 * self.alpha + self.n_samples)
            parent_counts = np.zeros(2)
            parent_counts[0] = np.sum(self.data[:, parent] == 0) + 2 * self.alpha
            parent_counts[1] = np.sum(self.data[:, parent] == 1) + 2 * self.alpha
            parent_probs = parent_counts / (4 * self.alpha + self.n_samples)
            probs = np.zeros((2, 2))
            probs[0, 0] = joint_probs[0, 0] / parent_probs[0]
            probs[0, 1] = joint_probs[0, 1] / parent_probs[0]
            probs[1, 0] = joint_probs[1, 0] / parent_probs[1]
            probs[1, 1] = joint_probs[1, 1] / parent_probs[1]
            self.log_theta[i] = np.log(probs)

    def get_tree(self):
        return self.parents.copy()

    def get_log_params(self):
        log_params = np.zeros((self.n_vars, 2, 2))
        for i in range(self.n_vars):
            if i != self.root:
                log_params[i] = self.log_theta[i]
        log_params[self.root, 0, 0] = self.log_root_probs[0]
        log_params[self.root, 0, 1] = self.log_root_probs[1]
        log_params[self.root, 1, 0] = self.log_root_probs[0]
        log_params[self.root, 1, 1] = self.log_root_probs[1]
        return log_params

    def _log_prob_exhaustive(self, row):
        obs = ~np.isnan(row)
        miss = np.where(~obs)[0]
        if miss.size == 0:
            return self._compute_tree_log_probs(row.reshape(1, -1))[0]
        acc = []
        for comb in itertools.product((0, 1), repeat=miss.size):
            r = row.copy()
            r[miss] = comb
            acc.append(self._compute_tree_log_probs(r.reshape(1, -1))[0])
        return logsumexp(acc)

    def _log_prob_sumprod(self, row):
        log_params = self.get_log_params()
        obs = {i: int(row[i]) for i in range(self.n_vars) if not np.isnan(row[i])}

        def pass_up(j):
            msgs = [pass_up(c) for c in self.children[j]]
            out = np.full(2, -np.inf)
            for pv in (0, 1):
                tmp = []
                for xj in (0, 1):
                    val = log_params[j, pv, xj]
                    for m in msgs:
                        val += m[xj]
                    if j in obs and obs[j] != xj:
                        val = -np.inf
                    tmp.append(val)
                out[pv] = logsumexp(tmp)
            return out

        children_msgs = [pass_up(c) for c in self.children[self.root]]
        if self.root in obs:
            xr = obs[self.root]
            res = log_params[self.root, 0, xr] + sum(m[xr] for m in children_msgs)
        else:
            tmp = []
            for xr in (0, 1):
                val = log_params[self.root, 0, xr]
                for m in children_msgs:
                    val += m[xr]
                tmp.append(val)
            res = logsumexp(tmp)
        return res

    def log_prob(self, X, exhaustive: bool = False):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            if exhaustive:
                out[i] = self._log_prob_exhaustive(row)
            else:
                out[i] = self._log_prob_sumprod(row)
        return out

    def _compute_tree_log_probs(self, X):
        log_params = self.get_log_params()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        lp = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            val = log_params[self.root, 0, int(row[self.root])]
            for j in range(self.n_vars):
                if j == self.root:
                    continue
                p = self.parents[j]
                val += log_params[j, int(row[p]), int(row[j])]
            lp[i] = val
        return lp

    def sample(self, n_samples: int):
        log_params = self.get_log_params()
        order = [self.root]
        queue = [self.root]
        visited = {self.root}
        while queue:
            node = queue.pop(0)
            for child in self.children[node]:
                if child not in visited:
                    order.append(child)
                    visited.add(child)
                    queue.append(child)
        S = np.zeros((n_samples, self.n_vars))
        for t in range(n_samples):
            for j in order:
                if j == self.root:
                    probs = np.exp(log_params[j, 0])
                    S[t, j] = np.random.choice((0, 1), p=probs)
                else:
                    parent_val = int(S[t, self.parents[j]])
                    probs = np.exp(log_params[j, parent_val])
                    S[t, j] = np.random.choice((0, 1), p=probs)
        return S

    def visualize_tree(self):
        """
        Print a graphical representation of the tree to the console.
        """
        print(f"Tree structure rooted at node {self.root}:")
        print_tree(self.root, self.parents, self.children)
if __name__ == "__main__":
    data = load_dataset("datasets/nltcs/nltcs.train.data")
    model = BinaryCLT(data)
    print("Task 2a:")
    print(f"Tree: {model.get_tree()}")
    model.visualize_tree()
    print("\nTask 2b:")
    print(f"Log parameters shape: {model.get_log_params().shape}")
    print("Task 2c:")
    print(model.log_prob(data[0]))
    
    # Test marginal queries
    print("\nTask 2c (with marginal queries):")
    # Create some test queries with missing values - using the same number of variables as the model
    n_vars = model.n_vars
    test_queries = np.zeros((3, n_vars))
    test_queries[0, :] = data[0]  # First sample from the dataset
    
    # Create a marginal query with some missing values
    test_queries[1, :] = np.nan  # Set all to NaN first
    test_queries[1, 1] = 0.0     # Set some observed values
    test_queries[1, 4] = 1.0
    
    # Another marginal query
    test_queries[2, :] = np.nan  # Set all to NaN first
    test_queries[2, 1] = 0.0     # Set some observed values
    test_queries[2, 2] = 1.0
    test_queries[2, 3] = 1.0
    
    # Test with efficient inference
    print("Efficient inference:")
    log_probs_efficient = model.log_prob(test_queries, exhaustive=False)
    print(log_probs_efficient)
    
    # Test with exhaustive inference for small number of missing variables
    print("Exhaustive inference (first query only):")
    log_probs_exhaustive = model.log_prob(test_queries[0:1], exhaustive=True)
    print(log_probs_exhaustive)
    
    # Test that the sum of probabilities of all possible states equals 1
    print("\nVerifying normalization (sum of all probabilities = 1):")
    
    # For small models (few variables), we can enumerate all possible states
    if model.n_vars <= 16:  # Only do this for reasonably sized models
        # Generate all possible binary configurations
        all_possible_states = np.array(list(itertools.product([0, 1], repeat=model.n_vars)))
        
        # Compute log probabilities for all states
        log_probs_all_states = model.log_prob(all_possible_states, exhaustive=False)
        
        # Convert to probabilities and sum
        probs_sum = np.sum(np.exp(log_probs_all_states))
        
        print(f"Sum of probabilities of all 2^{model.n_vars} states: {probs_sum}")
        print(f"Difference from 1.0: {abs(probs_sum - 1.0)}")
        
        # Check if it's close to 1 (allowing for numerical precision issues)
        if abs(probs_sum - 1.0) < 1e-10:
            print("✓ Distribution is properly normalized!")
        else:
            print("✗ Distribution is NOT properly normalized!")
    else:
        print(f"Model has {model.n_vars} variables, too many to enumerate all 2^{model.n_vars} states.")
    
    print("Task 2d:")
    print(f"Samples: {model.sample(3)}")
    
    # Task 2e: Learn CLT with specific parameters and evaluate
    def evaluate_clt_on_nltcs():
        print("\nTask 2e:")
        # Load datasets
        train_data = load_dataset("datasets/nltcs/nltcs.train.data")
        test_data = load_dataset("datasets/nltcs/nltcs.test.data")
        
        # Learn CLT with root=0 and alpha=0.01
        model = BinaryCLT(train_data, root=0, alpha=0.01)
        
        # Print tree structure
        print("Tree structure:")
        model.visualize_tree()
        
        # Print CPTs (log parameters)
        log_params = model.get_log_params()
        print("\nLog parameters (CPTs) shape:", log_params.shape)
        
        # Calculate average log-likelihoods
        train_log_probs = model.log_prob(train_data)
        test_log_probs = model.log_prob(test_data)
        
        avg_train_ll = np.mean(train_log_probs)
        avg_test_ll = np.mean(test_log_probs)
        
        print(f"\nAverage train log-likelihood: {avg_train_ll:.6f}")
        print(f"Average test log-likelihood: {avg_test_ll:.6f}")
    
    # Run the evaluation
    evaluate_clt_on_nltcs()
    
    # Task: Compare inference methods on marginal queries
    def compare_inference_methods():
        print("\nComparing inference methods on marginal queries:")
        
        # Load training data and train the model
        print("Loading training data...")
        train_data = load_dataset("datasets/nltcs/nltcs.train.data")
        
        print("Training BinaryCLT model...")
        model = BinaryCLT(train_data, root=0, alpha=0.01)
        
        # Load marginal queries
        print("Loading marginal queries...")
        marginal_queries = load_dataset("datasets/nltcs/nltcs_marginals.data")
        print(f"number of marginal queries: {len(marginal_queries)}")
        
        # Take a subset of queries for exhaustive inference (which can be slow)
        num_queries_for_exhaustive = 10
        subset_queries = marginal_queries[:num_queries_for_exhaustive]
        
        print(f"Computing log probabilities for {len(marginal_queries)} queries using efficient inference...")
        start_efficient = time.time()
        efficient_log_probs = model.log_prob(marginal_queries, exhaustive=False)
        end_efficient = time.time()
        
        print(f"Computing log probabilities for {len(subset_queries)} queries using efficient inference...")
        start_efficient = time.time()
        efficient_log_probs = model.log_prob(subset_queries, exhaustive=False)
        end_efficient = time.time()
        
        print(f"Computing log probabilities for {len(subset_queries)} queries using exhaustive inference...")
        start_exhaustive = time.time()
        exhaustive_log_probs = model.log_prob(subset_queries, exhaustive=True)
        end_exhaustive = time.time()

        samples = model.sample(1000)
        avg_ll_generated = model.log_prob(samples).mean()
        print("Average log-likelihood of generated samples:", avg_ll_generated)
        
        # Compare results for the subset
        efficient_subset = efficient_log_probs[:num_queries_for_exhaustive]
        
        # Calculate differences
        differences = np.abs(efficient_subset - exhaustive_log_probs)
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)
        
        print("\nResults:")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        print(f"\nEfficient inference runtime: {end_efficient - start_efficient:.4f} seconds")
        print(f"Exhaustive inference runtime (subset): {end_exhaustive - start_exhaustive:.4f} seconds")
        
        # Check if they are effectively the same (allowing for numerical precision)
        if max_diff < 1e-5:
            print("\nConclusion: Both inference methods deliver the same results within numerical precision.")
        else:
            print("\nConclusion: The inference methods produce different results.")
            
        # Print some examples
        print("\nExample comparisons (first 10 queries):")
        print("Query | Efficient | Exhaustive | Difference")
        print("-" * 50)
        for i in range(min(10, len(subset_queries))):
            print(f"{i+1} | {efficient_subset[i]:.6f} | {exhaustive_log_probs[i]:.6f} | {differences[i]:.6f}")
    
    # Run the comparison of inference methods
    compare_inference_methods()

    