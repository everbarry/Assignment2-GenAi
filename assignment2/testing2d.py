from ex2_jimena import load_dataset, BinaryCLT  # Replace with actual filename
import numpy as np

if __name__ == "__main__":
    # Load data and initialize model
    data = load_dataset("datasets/nltcs/nltcs.train.data")
    model = BinaryCLT(data, root=0, alpha=0.01)

    print("\n--- Test: Ancestral Sampling Output ---")

    # 1. Generate a few samples
    samples = model.sample(5)
    print("\nGenerated Samples (5):")
    for s in samples:
        print(s.astype(int))

    # 2. Check shape and value sanity
    print("\nShape of samples:", samples.shape)
    print("Unique values in samples:", np.unique(samples))

    # 3. Estimate marginal probabilities from many samples
    big_samples = model.sample(1000)
    marginal_probs = np.mean(big_samples, axis=0)
    print("\nEstimated marginal probabilities (from 1000 samples):")
    print(np.round(marginal_probs, 3))

    # 4. Compute likelihoods of generated samples
    log_probs = model.log_prob(samples)
    print("\nLog-likelihoods of the 5 generated samples:")
    print(np.round(log_probs, 4))
