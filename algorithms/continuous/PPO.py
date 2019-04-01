import torch

from collections import deque
from common.abstract.base_agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, 
        env, 
        actor, critic,
        actor_optim, critic_optim,
        device,
        gamma,
        td_lambda,
        epsilon,
        gae,
        ):
        super().__init__()

        self.env = env

        self.device = device
        self.actor, self.critic = actor, critic
        self.actor_optim, self.critic_optim = actor_optim, critic_optim

        self.gamma = gamma
        self.td_lambda = td_lambda
        self.epsilon = epislon
        self.gae = gae

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        dist = self.actor.forward(state)
        value = self.cr


    def train_model(self, last_value, done):

        torch.cat(self.log_probs)
        torch.cat(self.values + [last_value])
        torch.FloatTensor(self.dones + [done]).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)

        # Compute GAE
        length = len(rewards)
        R = 0
        torch.zeros(length).to(self.device)
        for i in reversed(range(length)):
            R = rewards[i] + (self.gamma * self.td_lambda * R)

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

            steps = 0
            done = False
            while not done:
                steps += 1
                action
                action, value, log_prob, entropy = self.select_action(state)

                next_state,
