import numpy as np
import csv
import time
import os
from ex2 import BinaryCLT

def load_dataset(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        dataset = np.array(list(reader)).astype(np.float32)
    return dataset

def evaluate_model(model, test_data):
    # Compute average log likelihood
    log_probs = model.log_prob(test_data)
    avg_log_likelihood = np.mean(log_probs)
    return avg_log_likelihood

def test_dataset(dataset_name, datasets_dir):
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    
    # Load training, validation, and test data
    train_data = load_dataset(os.path.join(datasets_dir, dataset_name, f"{dataset_name}.train.data"))
    valid_data = load_dataset(os.path.join(datasets_dir, dataset_name, f"{dataset_name}.valid.data"))
    test_data = load_dataset(os.path.join(datasets_dir, dataset_name, f"{dataset_name}.test.data"))
    
    print(f"Training samples: {train_data.shape[0]}, Variables: {train_data.shape[1]}")
    print(f"Validation samples: {valid_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Train the model
    print("\nTraining BinaryCLT model...")
    start_time = time.time()
    model = BinaryCLT(train_data)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get the learned tree structure
    tree = model.get_tree()
    print(f"\nLearned tree structure (parent of each node):")
    print(tree)
    
    # Evaluate on training, validation and test sets
    print("\nEvaluating model...")
    train_log_likelihood = evaluate_model(model, train_data)
    valid_log_likelihood = evaluate_model(model, valid_data)
    test_log_likelihood = evaluate_model(model, test_data)
    
    print(f"Average log-likelihood on training set: {train_log_likelihood:.4f}")
    print(f"Average log-likelihood on validation set: {valid_log_likelihood:.4f}")
    print(f"Average log-likelihood on test set: {test_log_likelihood:.4f}")
    
    # Generate a sample
    print("\nSample from the model:")
    sample = model.sample(1)[0]
    print(sample)
    
    return train_log_likelihood, valid_log_likelihood, test_log_likelihood

def main():
    # Get the correct path to the datasets folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    datasets_dir = os.path.join(parent_dir, "datasets")
    
    # List of datasets to test
    datasets = ["nltcs", "mushrooms", "dna"]
    
    results = {}
    
    for dataset in datasets:
        try:
            train_ll, valid_ll, test_ll = test_dataset(dataset, datasets_dir)
            results[dataset] = (train_ll, valid_ll, test_ll)
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
    
    # Print summary of results
    print("\n\n" + "="*80)
    print("Summary of Results")
    print("="*80)
    print(f"{'Dataset':<15} {'Training LL':<20} {'Validation LL':<20} {'Test LL':<20}")
    print("-"*80)
    
    for dataset, (train_ll, valid_ll, test_ll) in results.items():
        print(f"{dataset:<15} {train_ll:<20.4f} {valid_ll:<20.4f} {test_ll:<20.4f}")

if __name__ == "__main__":
    main() 