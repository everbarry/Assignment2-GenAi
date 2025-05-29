import numpy as np
import itertools
from ex2 import BinaryCLT

def test_inference_methods_simple_tree():
    print("\nTesting inference methods with a simple 3-node tree:")
    
    # Create a simple dataset with 3 variables
    # This will create a simple tree: 0 -> 1 -> 2
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    data = np.zeros((n_samples, 3))
    
    # Generate data with dependencies
    # Variable 0 is root (random)
    data[:, 0] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Variable 1 depends on 0
    for i in range(n_samples):
        if data[i, 0] == 0:
            data[i, 1] = np.random.choice([0, 1], p=[0.8, 0.2])
        else:
            data[i, 1] = np.random.choice([0, 1], p=[0.3, 0.7])
    
    # Variable 2 depends on 1
    for i in range(n_samples):
        if data[i, 1] == 0:
            data[i, 2] = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            data[i, 2] = np.random.choice([0, 1], p=[0.2, 0.8])
    
    # Create model with fixed root
    model = BinaryCLT(data, root=0, alpha=0.01)
    
    # Visualize the tree
    print("Tree structure:")
    model.visualize_tree()
    
    # Create all possible marginal queries
    # For 3 variables, we have 2^3 = 8 possible configurations
    all_configs = np.array(list(itertools.product([0, 1], repeat=3)))
    
    # Create marginal queries by masking some variables
    test_queries = []
    
    # Query 1: Only variable 0 observed
    q1 = np.array([0, np.nan, np.nan])
    test_queries.append(q1)
    
    # Query 2: Only variable 1 observed
    q2 = np.array([np.nan, 1, np.nan])
    test_queries.append(q2)
    
    # Query 3: Variables 0 and 2 observed
    q3 = np.array([1, np.nan, 0])
    test_queries.append(q3)
    
    # Query 4: All variables observed
    q4 = np.array([0, 1, 0])
    test_queries.append(q4)
    
    # Convert to numpy array
    test_queries = np.array(test_queries)
    
    # Compute log probabilities using both methods
    print("\nComputing log probabilities:")
    efficient_log_probs = model.log_prob(test_queries, exhaustive=False)
    exhaustive_log_probs = model.log_prob(test_queries, exhaustive=True)
    
    # Compare results
    print("\nResults comparison:")
    print("Query | Variables Observed | Efficient | Exhaustive | Difference")
    print("-" * 70)
    
    for i, query in enumerate(test_queries):
        observed = [f"X{j}={int(val)}" for j, val in enumerate(query) if not np.isnan(val)]
        observed_str = ", ".join(observed)
        diff = abs(efficient_log_probs[i] - exhaustive_log_probs[i])
        print(f"{i+1} | {observed_str:<20} | {efficient_log_probs[i]:.6f} | {exhaustive_log_probs[i]:.6f} | {diff:.6f}")
    
    # Verify that the sum of probabilities for all configurations equals 1
    print("\nVerifying normalization:")
    log_probs_all = model.log_prob(all_configs, exhaustive=False)
    probs_sum = np.sum(np.exp(log_probs_all))
    print(f"Sum of probabilities for all 2^3 = 8 configurations: {probs_sum:.10f}")
    
    # Calculate the difference between efficient and exhaustive methods
    diff_sum = np.sum(np.abs(efficient_log_probs - exhaustive_log_probs))
    print(f"\nTotal absolute difference between methods: {diff_sum:.10f}")
    
    if diff_sum < 1e-5:
        print("✓ Both inference methods produce the same results within numerical precision!")
    else:
        print("✗ The inference methods produce different results!")
        print("Debugging the implementation may be necessary.")

if __name__ == "__main__":
    test_inference_methods_simple_tree() 