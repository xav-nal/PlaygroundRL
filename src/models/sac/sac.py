#
# implementation source: https://github.com/higgsfield-ai/higgsfield/blob/main/higgsfield/rl/rl_adventure_2/7.soft%20actor-critic.ipynb
#
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from .networks import Actor, Critic, SoftQ
from .replay_buffer import ReplayBuffer

class SAC:
    def __init__(self, vec_env, cfg, device):
        self.device = device
        self.vec_env = vec_env
        self.action_dim = vec_env.action_space.shape[0]
        self.inputs_dim = vec_env.observation_space.shape[0]
        self.hidden_dim = cfg.HIDDEN_DIM
        self.batch_size = cfg.BATCH_SIZE
        self.max_steps = cfg.MAX_STEPS
        self.init_w = cfg.INIT_WEIGHTS
        self.gamma = cfg.GAMMA
        self.mean_lambda = cfg.MEAN_LAMBDA
        self.std_lambda = cfg.STD_LAMBDA
        self.z_lambda = cfg.Z_LAMBDA
        self.soft_tau = cfg.SOFT_TAU
        
        self.replay_buffer = ReplayBuffer(cfg.REPLAY_BUFFER)

        self.critic = Critic(self.inputs_dim, self.hidden_dim, self.init_w).to( self.device)
        self.softQ = SoftQ(self.inputs_dim, self.action_dim, self.hidden_dim, self.init_w).to( self.device)
        self.actor = Actor(self.inputs_dim, self.action_dim, cfg.ACTOR, self.device).to(self.device)

        self.target_critic = Critic(self.inputs_dim, self.hidden_dim, self.init_w).to( self.device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=cfg.LR_CRITIC)
        self.optimizer_softQ = optim.Adam(self.softQ.parameters(), lr=cfg.LR_SOFTQ)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=cfg.LR_ACTOR)

        self.value_criterion  = nn.MSELoss()
        self.softQ_criterion = nn.MSELoss()

        
    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

       
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)


        expected_q_value = self.softQ(state, action)
        expected_value   = self.critic(state)
        new_action, log_prob, z, mean, log_std = self.actor.evaluate(state)

        target_value = self.target_critic(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value

        q_value_loss = self.softQ_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.softQ(state, new_action)
        next_value = expected_new_q_value - log_prob
        critic_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        
        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss  = self.std_lambda  * log_std.pow(2).mean()
        z_loss    = self.z_lambda    * z.pow(2).sum(1).mean()

        actor_loss += mean_loss + std_loss + z_loss

        self.optimizer_softQ.zero_grad()
        q_value_loss.backward()
        self.optimizer_softQ.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

    def learn(self, total_timesteps):
        rewards = []

        for timestep in tqdm(range(1,total_timesteps)):
            state_np = self.vec_env.reset()
            state = torch.FloatTensor(state_np)
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.actor.get_action(state)
                next_state, reward, done, _ = self.vec_env.step(action)
                        
                self.replay_buffer.push(state.squeeze(), action.squeeze(), reward.squeeze(), next_state.squeeze(), done.squeeze())

                if len(self.replay_buffer) > self.batch_size:
                    self.update()
                
                state = next_state
                episode_reward += reward
                
                
                if done:
                    break

            rewards.append(episode_reward)

        
        return rewards

