import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, init_w):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(inputs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value
    

class SoftQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w):
        super(SoftQ, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

  
class Actor(nn.Module):
    def __init__(self, inputs_dim, num_actions, cfg, device):
        super(Actor, self).__init__()
        
        self.device = device

        self.hidden_dim = cfg.HIDDEN_DIM
        self.init_w = cfg.INIT_WEIGHTS
        self.log_std_min = cfg.LOG_STD_MIN
        self.log_std_max = cfg.LOG_STD_MAX
        self.epsilon = cfg.EPSILON
        
        self.fc1 = nn.Linear(inputs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.mean_linear = nn.Linear(self.hidden_dim, num_actions)
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.bias.data.uniform_(-self.init_w, self.init_w)
        
        self.log_std_linear = nn.Linear(self.hidden_dim, num_actions)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.bias.data.uniform_(-self.init_w, self.init_w)
        

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        a = normal.sample()
        action = torch.tanh(a)
        
        log_prob = normal.log_prob(a) - torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, a, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        a      = normal.sample()
        action = torch.tanh(a)
        
        action  = action.detach().cpu().numpy()
        return action[0]
    
