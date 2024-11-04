import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    def __init__(self, policy: nn.Module, logger=None, device=None):
        super().__init__()
        self.policy = policy

        self.n_inner_epochs = 3
        self.clamp_margin = 0.3
        self.entropy_coef = 0.05
        self.entropy_decay = 0.90
        self.value_coef = 0.5

        self.log_probs_history = []
        self.actions_history = []
        self.rewards_history = []
        self.values_history = []
        self.terminated_history = []
        self.state_history = []
        self.eval = False
        self.logger = logger
        self.device = device

    def act(self, obs):
        self.policy.eval()
        
        with torch.no_grad():
            action_logits, value = self.policy(obs)
 
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()

        if not self.eval:
            self.log_probs_history.append(action_distribution.log_prob(action).tolist())
            self.values_history.append(value.tolist())
            self.actions_history.append(action.tolist())
            self.state_history.append(obs.tolist())

        return action, value

    def update(self, optimizer):
        old_log_probs = torch.tensor(self.log_probs_history).to(self.device).detach().T.reshape(-1)
        old_values = torch.tensor(self.values_history).to(self.device).squeeze().detach().T.reshape(-1)
        old_actions = torch.tensor(self.actions_history).to(self.device).detach().T.reshape(-1)
        old_states = torch.tensor(self.state_history).to(self.device).detach()
        old_states = old_states.transpose(0, 1).reshape(-1, *old_states.shape[2:])

        rewards = torch.tensor(self.rewards_history).to(self.device).T.reshape(-1)
        terminated_history = torch.tensor(self.terminated_history).to(self.device).T.reshape(-1)
        
        advantages, returns = self.compute_gae(rewards, old_values, ~ terminated_history)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        dataset = torch.utils.data.TensorDataset(
                old_states,
                old_actions,
                old_log_probs,
                returns,
                advantages)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        self.policy.train()
        
        for _ in range(self.n_inner_epochs):
            for old_states, old_actions, old_log_probs, returns, advantages in dataloader:
                
                logits, state_values = self.policy(old_states)
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(old_actions)
                entropy = distribution.entropy().mean()
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = -ratio * advantages
                
                surr2 = -torch.clamp(ratio, 1 - self.clamp_margin, 1 + self.clamp_margin) * advantages
                
                policy_loss = torch.max(surr1, surr2).mean() 
                value_loss = (returns - state_values.squeeze()).pow(2).mean()
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                optimizer.zero_grad()

                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                if self.logger is not None:
                    self.logger.log({"loss": loss.item()})
                    self.logger.log({"policy_loss": policy_loss.item()})
                    self.logger.log({"value_loss": value_loss.item()})
                    self.logger.log({"entropy": entropy.item()})
                    self.logger.log({"grad_norm": total_norm})
                    self.logger.log({"surr1": surr1.mean()})
                    self.logger.log({"surr2": surr2.mean()})
                    self.logger.log({"ratio": ratio.mean()})
               
                optimizer.step()

        if self.entropy_coef > 0.001:
            self.entropy_coef *= self.entropy_decay
            
        self.log_probs_history = []
        self.rewards_history = []
        self.values_history = []
        self.terminated_history = []
        self.state_history = []
        self.actions_history = []

    def compute_gae(self, rewards, values, masks, gamma=0.99, lamda=0.9):
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            gae = delta + gamma * lamda * masks[t] * gae
            advantages[t] = gae
            next_value = values[t]
            returns[t] = advantages[t] + values[t]

        return advantages.detach(), returns.detach()
        
    def discount_rewards(self, rewards, terminated, gamma=0.99):

        discounted_rewards = torch.zeros(len(rewards))
        running_add = 0
        for t in reversed(range(len(rewards))):
            if terminated[t]:
                running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def reset(self):
        self.log_probs_history = []
        self.rewards_history = []
        self.values_history = []
        self.terminated_history = []
        self.state_history = []
        self.actions_history = []
