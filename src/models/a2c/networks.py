import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size, verbose=0):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.verbose = verbose

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.fc2(x)

        if(self.verbose):
            print("- Critic -")
            print("state value: ",state_value)

        return state_value


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, verbose=0):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.mean = nn.Linear(hidden_size, num_actions)
        self.std = nn.Linear(hidden_size, num_actions)

        self.verbose = verbose

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.mean(x)
        std = self.std(x)
        std = torch.clamp(std, min=1e-6, max=1)
        
        dist = torch.distributions.Normal(mean, std)

        action =  F.tanh(dist.rsample())
        log_pi = dist.log_prob(action)

        if(self.verbose):
            print("- Actor -")
            print("action mean: ", mean)
            print("action std: ", std)
            print("action min: ", torch.min(action))
            print("action max: ", torch.max(action))
            print("log pi min: ", torch.min(log_pi))
            print("log pi max: ", torch.max(log_pi))

        return action, log_pi
    
