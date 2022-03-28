import torch.nn as nn
import numpy as np


class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


def add_gradients(total_a2c_loss, kernels, particles, n_particles):
    total_a2c_loss.backward(retain_graph=True)

    for i in range(n_particles):
        for j in range(n_particles):
            if j == i:
                continue

            theta_i = particles[i]["prob_agent"].model.parameters()
            theta_j = particles[j]["prob_agent"].model.parameters()

            for (wi, wj) in zip(theta_i, theta_j):
                wi.grad = wi.grad + wj.grad * kernels[j, i].detach()
