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
        self.model = model
        self.optim = optim

        self.gamma = hyper_params["gamma"]
        self.entropy_rate = hyper_params["entropy_rate"]
        self.action_min = hyper_params["action_min"]
        self.action_max = hyper_params["action_max"]

        self.log_probs, self.entropies, self.rewards = [], [], []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist = self.model.forward(state)

        action = dist.sample().clamp(self.action_min, self.action_max)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.detach().cpu().numpy(), log_prob, entropy

    def append_samples(self, log_prob, entropy, reward):
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.rewards.append(reward)

    def compute_return(self):
        discounted_rewards = np.zeros(len(self.rewards))
        running_add = 0
        for i in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[i]
            discounted_rewards[i] = running_add
        return discounted_rewards

    def train_model(self):

        # R = torch.zeros(1, 1).cuda()
        # loss = 0
        # for i in reversed(range(len(self.rewards))):
        #     R = self.gamma * R + self.rewards[i]
        #     loss -= (self.log_probs[i] * R) \
        #              - (self.entropy_rate * self.entropies[i])

        # loss /= len(self.rewards)
        # loss = loss.squeeze()
        discounted_return = self.compute_return()

        log_probs = torch.cat(self.log_probs)
        entropies = torch.cat(self.entropies)
        discounted_return = torch.FloatTensor(
            discounted_return).to(self.device)

        loss = - log_probs * discounted_return + self.entropy_rate * entropies
        loss = loss.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.log_probs, self.entropies, self.rewards = [], [], []

    def write_log(self, score, recent_score, episode, total_episode):

        sys.stdout.write(f"[INFO] episode ({episode}/{total_episode}) | "
                         f"score {score:4.0f} | "
                         f"recent score {recent_score:4.1f}\n")

        wandb.log(
            {
                "recent score": recent_score,
                "score": score,
            }
        )

    def save_params(self, file_name):
        params = {
            "model_state_dict": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        super().save_params(file_name, params)

    def load_params(self, file_name, run_path=None):
        params = super().load_params(file_name, run_path)

        self.model.load_state_dict(params["model_state_dict"])
        self.optim.load_state_dict(params["optimizer"])

    def train(self, config):
        recent_scores = deque(maxlen=config['score_deque_len'])

        for episode in range(config['n_episode']):
            state = self.env.reset()

            score = 0
            done = False
            while not done:

                action, log_prob, entropy = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                self.append_samples(log_prob, entropy, reward)

                score += reward
                state = next_state

            recent_scores.append(score)

            self.train_model()

            self.write_log(score, np.mean(recent_scores),
                           episode, config['n_episode'])

    def test(self, config):
        for episode in range(config['n_episode']):
            state = self.env.reset()

            score = 0
            done = False
            while not done:
                action, _, _ = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                score += reward
                state = next_state

                if config['render']:
                    self.env.render()

            print(f"{score}")
