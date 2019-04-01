import hues
import wandb
import torch
import numpy as np
from collections import deque

from common.abstract.base_agent import BaseAgent


class GAE(BaseAgent):
    def __init__(
        self,
        env,
        actor, critic,
        actor_optim, critic_optim,
        device,
        gamma=0.99,
        td_lambda=0.95,
        entropy_rate=0.001,
        batch_size=16,
        action_space_low=-2,
        action_space_high=2
    ):
        self.env = env

        self.device = device
        self.actor, self.critic = actor, critic
        self.actor_optim, self.critic_optim = actor_optim, critic_optim

        # Hyperparameters
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.ent_rate = entropy_rate
        self.batch_size = batch_size
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        # Memories
        self.values, self.log_probs, self.rewards, self.dones = [], [], [], []
        self.entropy = torch.FloatTensor([0]).to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        dist = self.actor.forward(state)
        value = self.critic.forward(state)

        action = dist.sample()
        clip_act = action.clamp(self.action_space_low, self.action_space_high)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return clip_act.detach().cpu().numpy(), value, log_prob, entropy

    def append_memory(self, value, log_prob, entropy, reward, done):
        self.entropy += entropy
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def train_model(self, last_state, done):

        last_state = torch.FloatTensor(last_state).to(self.device)
        last_value = self.critic.forward(last_state)

        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values + [last_value])
        dones = torch.FloatTensor(self.dones + [done]).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)

        delta = rewards + self.gamma * values[1:] * (1 - dones[1:]) \
            - values[:-1]

        length = len(delta)
        R = 0
        advantage = torch.zeros(length).to(self.device)
        for i in reversed(range(length)):
            R = delta[i] + (self.gamma * self.td_lambda * R)
            advantage[i] = R + values[i]

        # 그저 value를 더하고 뺀 것 뿐이지만
        # 이 계산이 들어가야 Backpropagation이 가능해진다.
        advantage -= values[:-1]

        actor_loss = -(log_probs * advantage.detach()) + \
            (self.ent_rate * self.entropy)
        actor_loss = actor_loss.mean()
        critic_loss = advantage.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.values, self.log_probs, self.rewards, self.dones = [], [], [], []
        self.entropy = torch.FloatTensor([0]).to(self.device)

    def write_log(self, score, recent_score, episode, total_episode):

        hues.info(f"episode ({episode}/{total_episode}) | "
                  f"score {score:4.0f} | recent score {recent_score:4.1f}")

        wandb.log(
            {
                "recent score": recent_score,
                "score": score,
            }
        )

    def train(self, **kargs):
        recent_scores = deque(maxlen=kargs['score_deque_len'])

        for episode in range(kargs['n_episode']):
            state = self.env.reset()

            score = 0
            steps = 0
            done = False
            while not done:
                if kargs['render']:
                    self.env.render()

                steps += 1
                action, value, log_prob, entropy = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action)

                self.append_memory(value, log_prob, entropy, reward, done)

                if steps >= self.batch_size or done:
                    self.train_model(next_state, done)
                    steps = 0

                score += reward
                state = next_state

            recent_scores.append(score)

            self.write_log(
                score,
                np.mean(recent_scores),
                episode,
                kargs['n_episode'])

    def test(self, **kargs):
        for episode in range(kargs['n_episode']):
            state = self.env.reset()

            score = 0
            done = False
            while not done:
                action, _, _, _ = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action)

                score += reward
                state = next_state

                if kargs['render']:
                    self.env.render()

            hues.info(f"episode {episode}, score {score}")
