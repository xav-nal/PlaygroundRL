

class ReplayBuffer():
    def __init__(self,max_size=1000000):
        self.data = []

    def __len__(self):
        return self.data.shape[0]
    
    def add_rollout(self, rollout):
        self.datat.append(rollout)


    def sample_random_data(self,batch_size):
        

