import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_evaluation(model, dataloader, save_path):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            output, _ = model(inputs)
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    projected_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        indices = labels == i
        plt.scatter(projected_features[indices, 0], projected_features[indices, 1], label=str(i))
    plt.legend()
    plt.savefig(save_path)
    plt.close()