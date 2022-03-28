import torch, torch.nn as nn

"""
Function to index a 3d tensor by a 2D tensor
Useful for calculating the A2C loss
"""


def _index(tensor_3d, tensor_2d):
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)

    return v

def get_parameters(nn_list):
    params = []
    for nn in nn_list:
        l = list(nn.parameters())

        l_flatten = [torch.flatten(p) for p in l]
        l_flatten = tuple(l_flatten)

        l_concat = torch.cat(l_flatten)

        params.append(l_concat)

    return torch.stack(params)

def compute_gradients_norms(particles, logger, epoch):
    policy_gradnorm, critic_gradnorm = 0, 0

    for particle in particles:

        prob_params = particle["prob_agent"].model.parameters()
        critic_params = particle["critic_agent"].critic_model.parameters()

        for w_prob, w_critic in zip(prob_params, critic_params):
            if w_prob.grad != None:
                policy_gradnorm += w_prob.grad.detach().data.norm(2) ** 2

            if w_critic.grad != None:
                critic_gradnorm += w_critic.grad.detach().data.norm(2) ** 2

    policy_gradnorm, critic_gradnorm = (
        torch.sqrt(policy_gradnorm),
        torch.sqrt(critic_gradnorm),
    )

    logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)
    logger.add_log("Critic Gradient norm", critic_gradnorm, epoch)
