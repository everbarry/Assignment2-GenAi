from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv


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
        
    # Print the current node
    prefix = "└── " if is_last else "├── "
    print(f"{indent}{prefix}Node {node_names[root]}")
    
    # Prepare indentation for children
    child_indent = indent + ("    " if is_last else "│   ")
    
    # Print children
    for i, child in enumerate(children[root]):
        print_tree(child, parents, children, node_names, child_indent, i == len(children[root]) - 1)



class BinaryCLT:
    def __init__(self, data, root = None, alpha: float = 0.01):
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
        
        # Set root node
        if root is None:
            self.root = np.random.randint(0, self.n_vars)
        else:
            assert 0 <= root < self.n_vars, "Root must be in range(data.shape[1])"
            self.root = root
        
        # Learn tree structure using mutual information
        self._learn_structure()
        
        # Compute parameters (CPTs)
        self._compute_parameters()
    
    def _learn_structure(self):
        """Learn the tree structure using the Chow-Liu algorithm."""
        # Compute pairwise mutual information
        mi_matrix = np.zeros((self.n_vars, self.n_vars))
        
        # Compute empirical marginals with Laplace smoothing
        # For each variable, we have 2 possible values, so we add 2*alpha to denominator
        marginal_counts = np.zeros((self.n_vars, 2))
        marginal_counts[:, 0] = np.sum(self.data == 0, axis=0) + self.alpha
        marginal_counts[:, 1] = np.sum(self.data == 1, axis=0) + self.alpha
        p_x = marginal_counts[:, 1] / (2 * self.alpha + self.n_samples)
        
        # Compute mutual information between all pairs of variables
        for i in range(self.n_vars):
            for j in range(i+1, self.n_vars):
                # Joint counts with Laplace smoothing
                counts_00 = np.sum((self.data[:, i] == 0) & (self.data[:, j] == 0)) + self.alpha
                counts_01 = np.sum((self.data[:, i] == 0) & (self.data[:, j] == 1)) + self.alpha
                counts_10 = np.sum((self.data[:, i] == 1) & (self.data[:, j] == 0)) + self.alpha
                counts_11 = np.sum((self.data[:, i] == 1) & (self.data[:, j] == 1)) + self.alpha
                
                # Joint probabilities
                p_00 = counts_00 / (self.n_samples + 4 * self.alpha)
                p_01 = counts_01 / (self.n_samples + 4 * self.alpha)
                p_10 = counts_10 / (self.n_samples + 4 * self.alpha)
                p_11 = counts_11 / (self.n_samples + 4 * self.alpha)
                
                # Marginal probabilities
                p_i0 = marginal_counts[i, 0] / (2 * self.alpha + self.n_samples)
                p_i1 = marginal_counts[i, 1] / (2 * self.alpha + self.n_samples)
                p_j0 = marginal_counts[j, 0] / (2 * self.alpha + self.n_samples)
                p_j1 = marginal_counts[j, 1] / (2 * self.alpha + self.n_samples)
                
                # Compute mutual information
                mi = 0
                if p_00 > 0: mi += p_00 * np.log(p_00 / (p_i0 * p_j0))
                if p_01 > 0: mi += p_01 * np.log(p_01 / (p_i0 * p_j1))
                if p_10 > 0: mi += p_10 * np.log(p_10 / (p_i1 * p_j0))
                if p_11 > 0: mi += p_11 * np.log(p_11 / (p_i1 * p_j1))
                
                # Store MI values (symmetric)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        # Negate MI for minimum spanning tree (since we want maximum MI)
        neg_mi_matrix = -mi_matrix
        
        # Find minimum spanning tree
        mst = minimum_spanning_tree(neg_mi_matrix).toarray()
        
        # Convert to directed tree rooted at self.root
        self.parents = np.full(self.n_vars, -1, dtype=int)
        self.children = [[] for _ in range(self.n_vars)]
        
        # Use BFS to direct edges away from root
        _, predecessors = breadth_first_order(mst, self.root, directed=False, return_predecessors=True)
        
        # Set parents and children based on BFS predecessors
        for i in range(self.n_vars):
            if i != self.root:
                parent = predecessors[i]
                self.parents[i] = parent
                self.children[parent].append(i)
    
    def _compute_parameters(self):
        """Compute the parameters (CPTs) of the tree."""
        # Store log parameters
        self.log_theta = np.zeros((self.n_vars, 2, 2))  # [var, parent_val, var_val]
        
        # Root node has no parent, compute marginal
        root_counts = np.zeros(2)
        root_counts[0] = np.sum(self.data[:, self.root] == 0) + self.alpha
        root_counts[1] = np.sum(self.data[:, self.root] == 1) + self.alpha
        root_probs = root_counts / (2 * self.alpha + self.n_samples)
        
        # Store log probabilities for root
        self.log_root_probs = np.log(root_probs)
        
        # For each non-root node, compute conditional probabilities
        for i in range(self.n_vars):
            if i != self.root:
                parent = self.parents[i]
                
                # Count occurrences with Laplace smoothing for joint probabilities
                counts = np.zeros((2, 2))  # [parent_val, i_val]
                
                # Count (parent=0, i=0)
                counts[0, 0] = np.sum((self.data[:, parent] == 0) & (self.data[:, i] == 0)) + self.alpha
                # Count (parent=0, i=1)
                counts[0, 1] = np.sum((self.data[:, parent] == 0) & (self.data[:, i] == 1)) + self.alpha
                # Count (parent=1, i=0)
                counts[1, 0] = np.sum((self.data[:, parent] == 1) & (self.data[:, i] == 0)) + self.alpha
                # Count (parent=1, i=1)
                counts[1, 1] = np.sum((self.data[:, parent] == 1) & (self.data[:, i] == 1)) + self.alpha
                
                # Calculate joint probabilities P(Y=y, Z=z)
                joint_probs = counts / (4 * self.alpha + self.n_samples)
                
                # Calculate marginal probabilities P(Z=z)
                parent_counts = np.zeros(2)
                parent_counts[0] = np.sum(self.data[:, parent] == 0) + 2 * self.alpha
                parent_counts[1] = np.sum(self.data[:, parent] == 1) + 2 * self.alpha
                parent_probs = parent_counts / (4 * self.alpha + self.n_samples)
                
                # Calculate conditional probabilities P(Y=y|Z=z) = P(Y=y,Z=z) / P(Z=z)
                probs = np.zeros((2, 2))
                probs[0, 0] = joint_probs[0, 0] / parent_probs[0]
                probs[0, 1] = joint_probs[0, 1] / parent_probs[0]
                probs[1, 0] = joint_probs[1, 0] / parent_probs[1]
                probs[1, 1] = joint_probs[1, 1] / parent_probs[1]
                
                # Store log probabilities
                self.log_theta[i] = np.log(probs)

    def get_tree(self):
        """
        Return the learned tree structure as a list of parent indices.
        If X_j is the parent of X_i then tree[i] = j.
        If X_i is the root of the tree then tree[i] = -1.
        """
        return self.parents.copy()

    def get_log_params(self):
        """
        Return the log parameters of the model.
        
        Returns:
            log_params: A D×2×2 array where log_params[i,j,k] = log p(xi = k|xτ(i) = j)
                        For the root node i, log_params[i,0,k] = log_params[i,1,k] = log p(xi = k)
        """
        log_params = np.zeros((self.n_vars, 2, 2))
        
        # Fill in parameters for non-root nodes (already computed in self.log_theta)
        for i in range(self.n_vars):
            if i != self.root:
                log_params[i] = self.log_theta[i]
        
        # For the root node, set both parent values to have the same probability
        log_params[self.root, 0, 0] = self.log_root_probs[0]
        log_params[self.root, 0, 1] = self.log_root_probs[1]
        log_params[self.root, 1, 0] = self.log_root_probs[0]
        log_params[self.root, 1, 1] = self.log_root_probs[1]
        
        return log_params
    
    def log_prob(self, x, exhaustive: bool = False):
        """
        Compute the log probability of a data point or batch of data points.
        Support for marginal queries where some values are missing (encoded as np.nan).
        
        Args:
            x: Data point or batch (n_samples x n_vars) with binary values (0, 1) or np.nan for missing values
            exhaustive: If True, compute using joint distribution (for testing)
                        If False, use efficient inference (variable elimination)
        
        Returns:
            Log probability array of shape (n_samples,)
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        log_probs = np.zeros(n_samples)
        
        # Get log parameters
        log_params = self.get_log_params()
        
        for i in range(n_samples):
            # Identify observed and missing variables
            observed_mask = ~np.isnan(x[i])
            observed_vars = np.where(observed_mask)[0]
            missing_vars = np.where(~observed_mask)[0]
            
            # If all variables are observed, use the original method
            if len(missing_vars) == 0:
                if exhaustive:
                    # Find the matching configuration in all possible configurations
                    all_configs = np.array(list(itertools.product([0, 1], repeat=self.n_vars)))
                    all_log_probs = self._compute_tree_log_probs(all_configs)
                    
                    # Normalize to get a proper distribution
                    log_norm = logsumexp(all_log_probs)
                    all_log_probs -= log_norm
                    
                    # Find the matching configuration
                    for j, config in enumerate(all_configs):
                        if np.array_equal(x[i], config):
                            log_probs[i] = all_log_probs[j]
                            break
                else:
                    # Use tree factorization for fully observed case
                    log_probs[i] = self._compute_tree_log_probs(x[i].reshape(1, -1))[0]
            else:
                # Handle marginal queries
                if exhaustive:
                    # Generate all possible configurations for missing variables
                    missing_configs = list(itertools.product([0, 1], repeat=len(missing_vars)))
                    
                    # Initialize accumulator for log sum
                    log_sum_probs = []
                    
                    # For each configuration of missing variables
                    for config in missing_configs:
                        # Create a complete sample with this configuration
                        complete_sample = x[i].copy()
                        for j, var_idx in enumerate(missing_vars):
                            complete_sample[var_idx] = config[j]
                        
                        # Compute log probability of this complete sample
                        log_prob = self._compute_tree_log_probs(complete_sample.reshape(1, -1))[0]
                        log_sum_probs.append(log_prob)
                    
                    # Compute log sum exp of all configurations
                    log_probs[i] = logsumexp(log_sum_probs)
                else:
                    # Efficient inference using variable elimination
                    log_probs[i] = self._variable_elimination(x[i], observed_vars, missing_vars, log_params)
        
        return log_probs
    
    def _variable_elimination(self, x, observed_vars, missing_vars, log_params):
        """
        Perform variable elimination for efficient inference.
        
        Args:
            x: Single data point with some missing values
            observed_vars: Indices of observed variables
            missing_vars: Indices of missing variables
            log_params: Log parameters of the model
            
        Returns:
            Log probability of the observed values
        """
        # If no missing variables, use the tree factorization
        if len(missing_vars) == 0:
            return self._compute_tree_log_probs(x.reshape(1, -1))[0]
        
        # Create a mapping from original variable indices to observed values
        observed_values = {}
        for var in observed_vars:
            observed_values[var] = int(x[var])
        
        # Perform message passing from leaves to root and then from root to leaves
        # First, identify all nodes in the subtree containing observed variables
        relevant_nodes = set(observed_vars)
        for var in observed_vars:
            # Add path from observed variable to root
            node = var
            while node != -1:  # -1 is the parent of root
                relevant_nodes.add(node)
                node = self.parents[node]
        
        # For each missing variable that's relevant, marginalize over it
        log_prob = 0.0
        
        # Handle the root node separately
        if self.root in observed_values:
            # Root is observed
            root_val = observed_values[self.root]
            log_prob += log_params[self.root, 0, root_val]  # Root probability
        else:
            # Root is missing, marginalize over it
            log_prob += logsumexp([log_params[self.root, 0, 0], log_params[self.root, 0, 1]])
        
        # Handle non-root nodes
        for node in range(self.n_vars):
            if node != self.root:
                parent = self.parents[node]
                
                if node in observed_values:
                    # Node is observed
                    node_val = observed_values[node]
                    
                    if parent in observed_values:
                        # Both node and parent are observed
                        parent_val = observed_values[parent]
                        log_prob += log_params[node, parent_val, node_val]
                    else:
                        # Node is observed but parent is missing, marginalize over parent
                        log_prob += logsumexp([
                            log_params[node, 0, node_val] + (log_params[parent, 0, 0] if parent == self.root else 0),
                            log_params[node, 1, node_val] + (log_params[parent, 0, 1] if parent == self.root else 0)
                        ])
                elif node in relevant_nodes:
                    # Node is missing but relevant, marginalize over it
                    if parent in observed_values:
                        # Parent is observed
                        parent_val = observed_values[parent]
                        log_prob += logsumexp([log_params[node, parent_val, 0], log_params[node, parent_val, 1]])
                    else:
                        # Both node and parent are missing, marginalize over both
                        log_prob += logsumexp([
                            logsumexp([log_params[node, 0, 0], log_params[node, 0, 1]]) + (log_params[parent, 0, 0] if parent == self.root else 0),
                            logsumexp([log_params[node, 1, 0], log_params[node, 1, 1]]) + (log_params[parent, 0, 1] if parent == self.root else 0)
                        ])
        
        return log_prob
    
    def _compute_tree_log_probs(self, x):
        """Helper method to compute log probabilities using tree factorization."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        log_probs = np.zeros(n_samples)
        
        # Get log parameters
        log_params = self.get_log_params()
        
        for i in range(n_samples):
            # Root probability
            log_probs[i] = log_params[self.root, 0, int(x[i, self.root])]
            
            # Add child conditional probabilities
            for j in range(self.n_vars):
                if j != self.root:
                    parent = self.parents[j]
                    parent_val = int(x[i, parent])
                    j_val = int(x[i, j])
                    log_probs[i] += log_params[j, parent_val, j_val]
        
        return log_probs
    
    def sample(self, n_samples: int):
        """
        Generate samples from the distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Binary samples of shape (n_samples, n_vars)
        """
        samples = np.zeros((n_samples, self.n_vars), dtype=float)
        
        # Get log parameters
        log_params = self.get_log_params()
        
        # Topological ordering starting from root
        order = [self.root]
        visited = {self.root}
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            for child in self.children[node]:
                if child not in visited:
                    order.append(child)
                    visited.add(child)
                    queue.append(child)
        
        # Generate samples following topological order
        for i in range(n_samples):
            for node in order:
                if node == self.root:
                    # Sample root from its marginal
                    root_probs = np.exp([log_params[node, 0, 0], log_params[node, 0, 1]])
                    samples[i, node] = np.random.choice([0, 1], p=root_probs)
                else:
                    # Sample node conditional on its parent
                    parent = self.parents[node]
                    parent_val = int(samples[i, parent])
                    probs = np.exp([log_params[node, parent_val, 0], log_params[node, parent_val, 1]])
                    samples[i, node] = np.random.choice([0, 1], p=probs)
        
        return samples
    
    def visualize_tree(self):
        """
        Print a graphical representation of the tree to the console.
        """
        print(f"Tree structure rooted at node {self.root}:")
        print_tree(self.root, self.parents, self.children)
    
    
if __name__ == "__main__":
    data = load_dataset("../datasets/nltcs/nltcs.train.data")
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
        train_data = load_dataset("../datasets/nltcs/nltcs.train.data")
        test_data = load_dataset("../datasets/nltcs/nltcs.test.data")
        
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
        train_data = load_dataset("../datasets/nltcs/nltcs.train.data")
        
        print("Training BinaryCLT model...")
        model = BinaryCLT(train_data, root=0, alpha=0.01)
        
        # Load marginal queries
        print("Loading marginal queries...")
        marginal_queries = load_dataset("../tests/nltcs_marginals.data")
        print(f"number of marginal queries: {len(marginal_queries)}")
        
        # Take a subset of queries for exhaustive inference (which can be slow)
        num_queries_for_exhaustive = 10
        subset_queries = marginal_queries[:num_queries_for_exhaustive]
        
        print(f"Computing log probabilities for {len(marginal_queries)} queries using efficient inference...")
        efficient_log_probs = model.log_prob(marginal_queries, exhaustive=False)
        
        print(f"Computing log probabilities for {len(subset_queries)} queries using exhaustive inference...")
        exhaustive_log_probs = model.log_prob(subset_queries, exhaustive=True)
        
        # Compare results for the subset
        efficient_subset = efficient_log_probs[:num_queries_for_exhaustive]
        
        # Calculate differences
        differences = np.abs(efficient_subset - exhaustive_log_probs)
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)
        
        print("\nResults:")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        
        # Check if they are effectively the same (allowing for numerical precision)
        if max_diff < 1e-5:
            print("\nConclusion: Both inference methods deliver the same results within numerical precision.")
        else:
            print("\nConclusion: The inference methods produce different results.")
            
        # Print some examples
        print("\nExample comparisons (first 5 queries):")
        print("Query | Efficient | Exhaustive | Difference")
        print("-" * 50)
        for i in range(min(5, len(subset_queries))):
            print(f"{i+1} | {efficient_subset[i]:.6f} | {exhaustive_log_probs[i]:.6f} | {differences[i]:.6f}")
    
    # Run the comparison of inference methods
    compare_inference_methods()

    