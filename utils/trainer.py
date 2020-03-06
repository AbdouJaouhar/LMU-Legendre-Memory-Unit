import collections
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def generate_data(n_batches, length, split=0.5, seed=0,
                  predict_length=15, tau=17, washout=100, delta_t=1,
                  center=True):
    X = np.asarray(mackey_glass(sample_len=length+predict_length+washout, tau=tau,seed=seed, n_samples=n_batches))
    X = X[:, washout:, :]
    cutoff = int(split*n_batches)
    if center:
        X -= np.mean(X)  # global mean over all batches, approx -0.066
    Y = X[:, :-predict_length, :]
    X = X[:, predict_length:, :]
    assert X.shape == Y.shape
    return ((X[:cutoff], Y[:cutoff]),
            (X[cutoff:], Y[cutoff:]))

def cool_plot(X, Y, title=""):
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.scatter(X[:, 0], Y[:, 0] - X[:, 0], s=8, alpha=0.7, c=np.arange(X.shape[0]), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(X[:, 0], Y[:, 0] - X[:, 0], c='black', alpha=0.2)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t) - x(t)$")
    sns.despine(offset=15)

    plt.show()

def train(model, epochs, dataset, dataset_valid = None):
  for e in range(epochs):
      model.train()
      running_loss = 0
      with tqdm(total=len(dataset)) as bar:
        for i, (X, y) in enumerate(dataset):
            optimizer.zero_grad()

            output = model(X.cuda())

            loss_ll = criterion(output, y.cuda())

            loss_ll.backward()

            optimizer.step()

            running_loss += loss_ll.item()

            bar.update(1)
            bar.set_description("Epoch {} - Training loss: {}".format(e, running_loss/len(dataset)))

      model.eval()
      running_loss = 0
      outs = []
      with tqdm(total=len(dataset_valid)) as bar:
        for i, (X, y) in enumerate(dataset_valid):
            optimizer.zero_grad()

            output = model(X.cuda())

            loss_ll = criterion(output, y.cuda())

            running_loss += loss_ll.item()

            outs.append(output)
            bar.update(1)
            bar.set_description("Epoch {} - Training loss: {}".format(e, running_loss/len(dataset_valid)))
