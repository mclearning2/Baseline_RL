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

        self.states, self.actions, self.rewards = [], [], []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        policy = self.model.forward(state).detach().cpu().numpy()

        return np.random.choice(len(policy), 1, p=policy)[0]

    def append_samples(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_return(self):
        discounted_rewards = np.zeros(len(self.rewards))
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train_model(self):
        discounted_return = self.compute_return()

        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        torch.FloatTensor(self.rewards).to(self.device)
        discounted_return = torch.FloatTensor(
            discounted_return).to(self.device)

        log_policy = torch.log(self.model.forward(states) + 1e-5)

        acted_log_prob = log_policy.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = -(acted_log_prob * discounted_return).sum()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.rewards = [], [], []

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
                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                self.append_samples(state, action, reward)

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
                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                score += reward
                state = next_state

                if config['render']:
                    self.env.render()

            print(f"{score}")
