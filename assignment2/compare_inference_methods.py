import numpy as np
import csv
import sys
sys.path.append('.')
from ex2 import BinaryCLT, load_dataset

def load_marginals(filename):
    """Load marginal queries from a CSV file."""
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        queries = np.array(list(reader)).astype(np.float32)
    return queries

def main():
    # Load training data and train the model
    print("Loading training data...")
    train_data = load_dataset("datasets/nltcs/nltcs.train.data")
    
    print("Training BinaryCLT model...")
    model = BinaryCLT(train_data, root=0, alpha=0.01)
    
    # Load marginal queries
    print("Loading marginal queries...")
    marginal_queries = load_marginals("tests/nltcs_marginals.data")
    
    # Take a subset of queries for exhaustive inference (which can be slow)
    num_queries_for_exhaustive = 100
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

if __name__ == "__main__":
    main() 