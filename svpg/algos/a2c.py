import torch as th

from salina.rl.functional import gae

from svpg.algos.algo import Algo


class A2C(Algo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gae = cfg.algorithm.gae
        self.discount_factor = cfg.algorithm.discount_factor
        self.T = self.n_steps * self.n_envs * cfg.algorithm.max_epochs

    def compute_critic_loss(self, reward, must_bootstrap, critic):
        # Compute TD error
        td = gae(critic, reward, must_bootstrap, self.discount_factor, self.gae)
        # Compute critic loss
        td_error = td**2
        critic_loss = td_error.mean()
        return critic_loss, td

    def compute_policy_loss(self, action_logprobs, td):
        policy_loss = action_logprobs[:-1] * td.detach()
        return policy_loss.mean()

    def compute_loss(self, epoch, verbose=True):
        total_critic_loss, total_entropy_loss, total_policy_loss = 0, 0, 0

        for pid in range(self.n_particles):
            # Extracting the relevant tensors from the workspace
            transition_workspace = self.train_workspaces[pid].get_transitions()

            critic, done, action_logp, reward, truncated = transition_workspace[
                "critic",
                "env/done",
                "action_logprobs",
                "env/reward",
                "env/truncated",
            ]
            entropy = self.train_workspaces[pid]["entropy"]

            # Move to device
            critic = critic.to(self.device)
            done = done.to(self.device)
            action_logp = action_logp.to(self.device)
            reward = reward.to(self.device)
            entropy = entropy.to(self.device)
            truncated = truncated.to(self.device)

            must_bootstrap = th.logical_or(~done[1], truncated[1])

            # Compute loss
            critic_loss, td = self.compute_critic_loss(reward, must_bootstrap, critic)
            total_critic_loss = total_critic_loss + critic_loss

            policy_loss = self.compute_policy_loss(action_logp, td)
            total_policy_loss = total_policy_loss - policy_loss

            entropy_loss = th.mean(entropy)
            total_entropy_loss = total_entropy_loss - entropy_loss

        if verbose:
            self.logger.log_losses(
                epoch, total_critic_loss, total_entropy_loss, total_policy_loss
            )

        # n_steps = np.full(self.n_particles, self.n_steps * self.n_env)
        return total_policy_loss, total_critic_loss, total_entropy_loss
