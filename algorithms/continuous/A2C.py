import sys
import wandb
import torch
import numpy as np

from collections import deque
from common.abstract.base_agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, env, model, optim, device, hyper_params):
        super().__init__()

        self.env = env

        self.device = device
        self.actor, self.critic = model
        self.actor_optim, self.critic_optim = optim

        self.gamma = hyper_params['gamma']
        self.entropy_rate = hyper_params['entropy_rate']
        self.action_min = hyper_params['action_min']
        self.action_max = hyper_params['action_max']

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        dist = self.actor.forward(state)

        action = dist.sample().clamp(self.action_min, self.action_max)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.detach().cpu().numpy(), log_prob, entropy

    def train_model(self, state, reward, next_state, log_prob, entropy):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(state).to(self.device)

        value = self.critic.forward(state)
        next_value = self.critic.forward(next_state)

        advantage = reward + self.gamma * next_value - value

        actor_loss = - log_prob * advantage.detach() \
            + self.entropy_rate * entropy
        critic_loss = advantage.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def write_log(self, score, recent_score, episode, total_episode):

        sys.stdout.write(
            f"[INFO] episode ({episode}/{total_episode}) | "
            f"score {score:4.0f} | recent score {recent_score:4.1f}\n")

        wandb.log(
            {
                "recent score": recent_score,
                "score": score,
            }
        )

    def save_params(self, file_name):
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
        }
        super().save_params(file_name, params)

    def load_params(self, file_name, run_path=None):
        params = super().load_params(file_name, run_path)

        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optimizer"])
        self.critic_optim.load_state_dict(params["critic_optimizer"])

    def train(self, config):
        recent_scores = deque(maxlen=config['score_deque_len'])

        for episode in range(config['n_episode']):
            state = self.env.reset()

            score = 0
            done = False
            while not done:
                action, log_prob, entropy = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)
                self.train_model(state, reward, next_state, log_prob, entropy)

                score += reward
                state = next_state

            recent_scores.append(score)

            self.write_log(score, np.mean(recent_scores),
                           episode, config['n_episode'])

    def test(self, config):
        for episode in range(config['n_episode']):
            state = self.env.reset()

            score = 0
            done = False
            while not done:
                action, _, _, _ = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                score += reward
                state = next_state

                if config['render']:
                    self.env.render()

            print(f"{score}")
