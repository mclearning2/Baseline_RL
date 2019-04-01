import sys
import wandb
import torch
import numpy as np
import torch.nn.functional as F

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
        self.batch_size = hyper_params['batch_size']
        self.n_workers = hyper_params['n_workers']

        self.states, self.actions, self.rewards, self.dones = [], [], [], []

    def select_action(self, states):
        states = torch.FloatTensor(states).to(self.device)
        policies = self.actor.forward(states).detach().cpu().numpy()
        if len(states.size()) > 1:
            return [np.random.choice(len(policy), 1, p=policy)[0]
                    for policy in policies]
        else:
            return np.random.choice(len(policies), 1, p=policies)[0]

    def append_memory(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def train_model(self, last_value):

        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        policies = self.actor.forward(states)
        values = self.critic.forward(states).squeeze()

        R = last_value
        discounted_returns = torch.zeros(self.batch_size, self.n_workers)\
            .to(self.device)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + (self.gamma * R) * (1 - dones[i])
            discounted_returns[i] = R

        advantage = discounted_returns - values

        action_probs = policies.gather(2, actions.unsqueeze(2)).squeeze()
        log_probs = torch.log(action_probs + 1e-7)

        entropy = log_probs * action_probs

        actor_loss = - log_probs * advantage.detach() \
            + self.entropy_rate * entropy
        actor_loss = actor_loss.mean()
        critic_loss = F.mse_loss(values, advantage)

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.states, self.actions, self.rewards, self.dones = [], [], [], []

    def write_log(self, score, recent_score, episode, total_episode):
        sys.stdout.write(f"[INFO] episode ({episode}/{total_episode}) | "
                         f"score " + str(score) + " | " +
                         f"score mean {np.mean(score)} | "
                         f"recent score {recent_score:4.1f}\n")

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

    def load_params(self, file_name, run_path):
        params = super().load_params(file_name, run_path)

        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optimizer"])
        self.critic_optim.load_state_dict(params["critic_optimizer"])

    def train(self, config):
        recent_scores = deque(maxlen=config['score_deque_len'])

        steps = 0
        episode = 0
        scores = np.zeros(self.n_workers)

        states = self.env.reset()
        while episode < config['n_episode']:
            steps += 1

            actions = self.select_action(states)
            next_states, rewards, dones, info = self.env.step(actions)

            self.append_memory(states, actions, rewards, dones)

            scores += np.array(rewards)
            for score in scores:
                recent_scores.append(score)

            if steps >= self.batch_size:
                last_states = torch.FloatTensor(next_states).to(self.device)
                last_values = self.critic.forward(last_states).squeeze()

                self.train_model(last_values)

                self.states, self.actions = [], []
                self.rewards, self.dones = [], []

                steps = 0

            states = next_states

            for i in range(len(dones)):
                if dones[i]:
                    self.write_log(scores, np.mean(recent_scores), episode,
                                   config['n_episode'])
                    episode += 1
                    scores[i] = 0

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
