```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = 1000
lr = 0.00001
n_samples = 5000
n_features = 2304
n_targets = 1
n_epochs = 100
n_runs = 10

for run in tqdm(range(n_runs)):

    # prepare data
    X = np.random.normal(size=(n_samples, n_features))
    y = np.random.normal(size=(n_samples, n_targets))

    # initialize parameters
    W = np.random.normal(size=(n_features, n_targets))
    b = np.random.normal(size=(n_targets, n_targets))

    # keep track of errors
    errors = []

    for epoch in tqdm(range(n_epochs)):

        # shuffle the data
        permutation_indices = np.random.permutation(X.shape[0])
        permuted_X = X[permutation_indices]
        permuted_y = y[permutation_indices]

        for batch_index in range(0, n_samples, batch_size):

            # prepare batches
            batch_X = permuted_X[batch_index:batch_index+batch_size]
            batch_y = permuted_y[batch_index:batch_index+batch_size]

            # forward pass
            y_pred = np.dot(batch_X, W) + b

            # compute error
            error = batch_y - y_pred

            # update
            W += batch_X.T.dot(error) * lr
            b += np.mean(batch_y - y_pred) * lr

            # bookkeeping
            errors.append(np.mean(np.abs(error)))

    # plot run
    plt.plot(errors/np.mean(errors))

# save plots
plt.savefig('plot.png')
```

