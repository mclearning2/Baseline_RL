import gym
import torch
import torch.nn.functional as F
import numpy as np
from common.abstract.base_agent import BaseAgent

from algorithms.utils.buffer import ReplayMemory, PrioritizedReplayMemory
from algorithms.utils.update import hard_update

class DQN(BaseAgent):
    def __init__(self, env, online_net, target_net, optim, device, hyper_params):
        self.env = env
        self.device = device
        if hyper_params['dueling_q']:
            self.online_net = DuelingNet(online_net)
            self.target_net = DuelingNet(target_net)
        else:
            self.online_net = online_net
            self.target_net = target_net            

        self.optim = optim

        self.hp = hyper_params
        
        self.epsilon = self.hp['eps_start']
        self.eps_decay = (self.hp['eps_start'] - self.hp['eps_end']) \
                        / self.hp['eps_decay_steps']

        if self.hp['prioritized']:
            self.beta = self.hp['beta_start']
            self.beta_update = (1 - self.hp['beta_start']) / self.hp['beta_increase_steps']
            self.memory = PrioritizedReplayMemory(self.hp['memory_size'], self.hp['alpha'])
        else:
            self.memory = ReplayMemory(self.hp['memory_size'])

        hard_update(online_net, target_net)

    def decay_epsilon(self):
        self.epsilon = max(self.hp['eps_end'], self.epsilon - self.eps_decay)

    def increase_beta(self):
        self.beta = min(1, self.beta + self.beta_update)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if np.random.rand() <= self.epsilon:
            action = self.env.random_action()
        else:
            state = torch.FloatTensor(state).to(self.device)

            action = self.online_net(state).argmax(1).cpu().numpy()
            
        return action
        
    def train_model(self):
        if self.hp['prioritized']:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.hp['batch_size'], self.beta)
            
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.hp['batch_size'])

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones.astype(np.uint8)).to(self.device)

        q = self.online_net(states)
        logits = q.gather(1, actions)

        if self.hp['double_q']:
            next_q = self.online_net(next_states)
            next_target_q = self.target_net(next_states)
            
            target_q = next_target_q.gather(1, next_q.max(1)[1].unsqueeze(1))
        else:
            target_q = self.target_net(next_states).max(1)[0].unsqueeze(1)

        target = rewards + (1-dones) * target_q * self.hp['discount_factor']
        
        loss = (logits - target.detach()).pow(2).squeeze() * weights
        prios = loss + 1e-5
        loss = loss.mean()
        #loss = F.smooth_l1_loss(logits, target.detach())

        self.optim.zero_grad()
        loss.backward()

        if self.hp['prioritized']:
            self.memory.update_priorities(indices, prios.detach().cpu().numpy())

        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optim.step()

    def train(self):
        state = self.env.reset()
    
        total_step = 0
        while not self.env.is_episode_done():            
            total_step += 1
            action = self.select_action(state)
            
            next_state, reward, done, info = self.env.step(action)

            self.memory.save(state, action, reward, next_state, done)

            state = next_state

            if total_step > self.hp['start_learning_step']:
                self.increase_beta()
                self.decay_epsilon()
                self.train_model()

            if total_step % self.hp['target_update_period'] == 0:
                hard_update(self.online_net, self.target_net)

            if self.env.done[0]:
                self.write_log(
                    episode=self.env.episodes[0],
                    steps=self.env.steps[0],
                    score=self.env.scores[0],                    
                    recent_scores=np.mean(self.env.recent_scores),
                    epsilon=self.epsilon,
                    memory_size=len(self.memory),
                    total_step=total_step,
                )

class DuelingNet:
    def __init__(self, net):
        self.net = net

    def __call__(self, state):
        advantage, value = self.net(state)

        return value + advantage - advantage.mean()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    
    @property
    def state_dict(self):
        return self.net.state_dict

    def parameters(self):
        return self.net.parameters()