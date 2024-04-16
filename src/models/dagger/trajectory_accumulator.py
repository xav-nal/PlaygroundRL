import numpy as np
import abc


class TrajectoryAccumulator(abc.ABC):
    """
    Trajectory accumulator
    """
    def __init__(self, num_envs, num_obs, num_acts):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_acts = num_acts
        
        self.partial_trajectories = {env_id: [] for env_id in range(self.num_envs)}

        print("partial_trajectories", self.partial_trajectories)


    def add_step(self, env_id, step_dict):
        required_keys = {'acts', 'rews', 'next_obs', 'dones'}

        assert all(key in step_dict for key in required_keys)
        assert len(step_dict.keys()) == len(required_keys)

        self.partial_trajectories[env_id].append(step_dict)


    def get_partial_trajectories(self):
        return self.partial_trajectories
        

    def finish_trajectory(self, env_id):
        part_dicts = self.partial_trajectories[env_id]
        del self.partial_trajectories[env_id]

        return part_dicts
    

    def add_step_and_auto_finish(self, acts, rews, next_obs, dones):

        trajectories = []
        for env_idx in range(self.num_envs):
            assert env_idx in self.partial_trajectories.keys()


        for env_idx, (act, rew, next_ob, done) in enumerate(zip(acts, rews, next_obs, dones)):

            self.add_step(env_idx,
                        dict(acts=act,
                            rews=rew,
                            next_obs=next_ob,
                            dones=done,
                            ))
                
            if done:
                trajectory = self.finish_trajectory(env_idx)
                trajectories.append(trajectory)

        return trajectories


                
            
                

            

                
            




        

        