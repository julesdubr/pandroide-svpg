import torch as th

from svpg.algos.algo import Algo
from salina.rl.functionalb import gae


class A2C(Algo):
    def __init__(self, cfg, solo=False):
        super().__init__(cfg, solo)
        # self.gae = cfg.algorithm.gae
        self.discount_factor = cfg.algorithm.discount_factor
        self.gae_coef = cfg.algorithm.gae_coef
        self.T = self.n_steps * self.n_envs * cfg.algorithm.max_epochs

    def compute_critic_loss(self, reward, must_bootstrap, critic):
        # Compute temporal difference
        # target = reward[:-1] + self.discount_factor * critic[1:].detach() * (
        #     must_bootstrap.float()
        # )
        # td = target - critic[:-1]
        # assert (
        #     target.shape[1] == critic.shape[1]
        # ), f"Missing one element in the critic list: {target.shape} vs {critic.shape}"
        td = gae(critic, reward, must_bootstrap, self.discount_factor, self.gae_coef)

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()
        return critic_loss, td

    def compute_policy_loss(self, action_logprobs, td):
        policy_loss = action_logprobs[:-1] * td.detach()
        return policy_loss.mean()

    def compute_loss(self, epoch, verbose=True):
        total_critic_loss, total_entropy_loss, total_policy_loss = 0, 0, 0

        for train_workspace in self.train_workspaces:
            # Extracting the relevant tensors from the workspace
            transition_workspace = train_workspace.get_transitions()

            critic, done, action_logp, entrop, reward, truncated = transition_workspace[
                "critic",
                "env/done",
                "action_logprobs",
                "entropy",
                "env/reward",
                "env/truncated",
            ]

            must_bootstrap = th.logical_or(~done[1], truncated[1])

            # Compute loss
            critic_loss, td = self.compute_critic_loss(reward, must_bootstrap, critic)
            total_critic_loss = total_critic_loss + critic_loss

            policy_loss = self.compute_policy_loss(action_logp, td)
            total_policy_loss = total_policy_loss - policy_loss

            entropy_loss = th.mean(entrop)
            total_entropy_loss = total_entropy_loss - entropy_loss

        if verbose:
            self.logger.log_losses(
                epoch, total_critic_loss, total_entropy_loss, total_policy_loss
            )

        # n_steps = np.full(self.n_particles, self.n_steps * self.n_env)
        return total_policy_loss, total_critic_loss, total_entropy_loss
