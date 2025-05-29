import numpy as np
from ex2_jimena import load_dataset, BinaryCLT 

train = load_dataset("datasets/nltcs/nltcs.train.data")
model = BinaryCLT(train, root=0, alpha=0.01)

print("Tree parents:", model.get_tree())
print("Log-CPTs shape:", model.get_log_params().shape)



np.random.seed(36)
batch = train[np.random.choice(len(train), size=10, replace=False)]

#compute both efficient and exhaustive log probabilities
lp_eff      = model.log_prob(batch, exhaustive=False)
lp_exhaustive = model.log_prob(batch, exhaustive=True)

# print them  
print("Efficient inference:   ", lp_eff)
print("Exhaustive inference:  ", lp_exhaustive)

