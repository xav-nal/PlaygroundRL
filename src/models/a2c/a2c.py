import torch
import torch.optim as optim
import torch.nn.functional as F

import logging
from tqdm import tqdm
from datetime import datetime


from .networks import Actor,Critic


class A2C:
    def __init__(self, vec_env, cfg):
        self.cfg = cfg
        self.vec_env = vec_env
        self.num_actions = vec_env.action_space.shape[0]
        self.num_inputs = vec_env.observation_space.shape[0]
        
        self.actor = Actor( self.num_inputs, self.num_actions, self.cfg.HIDDEN_DIM)
        self.critic = Critic( self.num_inputs, self.cfg.HIDDEN_DIM)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.cfg.LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.cfg.LR_CRITIC)

        self.gamma = self.cfg.GAMMA


        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=f'src/models/logs/a2c_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.log', 
                            encoding='utf-8', level=logging.DEBUG)

        self.logger.info('----- A2C ------')
        self.logger.info('----- INIT ------')
        self.logger.info(f'num_inputs: {self.num_inputs}')
        self.logger.info(f'num_action: {self.num_actions}')
        self.logger.info(f'gamma: {self.gamma}')

        self.logger.info(f'lr actor: {self.cfg.LR_ACTOR}')
        self.logger.info(f'lr critic: {self.cfg.LR_CRITIC}')
        self.logger.info('')


    def online_update(self, next_state, state, action, reward, next_value, value, done, log_pi):
        delta = reward + self.gamma*next_value*(1-int(done)) - value

        actor_loss = -torch.mean(log_pi)*(delta)
        critic_loss = 0.5*delta**2

        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()     
        critic_loss.backward()
        self.optimizer_critic.step()

        self.logger.info('---- ONLINE UPDATE -----')
        self.logger.info(f'reward: {reward}')
        self.logger.info(f'action min: {torch.min(action)}')
        self.logger.info(f'action max: {torch.max(action)}')
        self.logger.info(f'value: {value}')
        self.logger.info(f'next_value: {next_value}')
        self.logger.info(f'done: {done}')
        self.logger.info(f'actor loss: {actor_loss}')
        self.logger.info(f'critic loss: {critic_loss}')
        self.logger.info('')

        return actor_loss + critic_loss


    def learn(self,total_timesteps):
        state_np = self.vec_env.reset()
        state = torch.FloatTensor(state_np)
        losses = []
        loss = 0

        for timestep in tqdm(range(1,total_timesteps)):
            action, log_pi = self.actor(state)

            next_state_np, reward, done, _ = self.vec_env.step(action.detach().numpy())
            next_state = torch.FloatTensor(next_state_np)

            value = self.critic(state)
            next_value = self.critic(next_state)

            loss += self.online_update(next_state, state, action, torch.FloatTensor(reward), next_value, value, done, log_pi)
            state = next_state

            if(done):
                done = False
                obs = self.vec_env.reset()
                losses.append(loss.item())
                loss = 0
                self.logger.info(f'END_EPISODE')

        return losses
