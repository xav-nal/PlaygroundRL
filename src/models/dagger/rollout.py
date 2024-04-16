
import numpy as np

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from .trajectory_accumulator import TrajectoryAccumulator
    
def generate_trajectories(
    policy: BasePolicy,
    venv: VecEnv ):
    """Generate trajectory dictionaries from a policy and an environment.
    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """

    trajectories = []
    trajectories_accum = TrajectoryAccumulator(venv.num_envs, venv.observation_space.shape[0], venv.action_space.shape[0])

    obs = venv.reset()

    init_acts = np.zeros((venv.num_envs, venv.action_space.shape[0]))
    init_rews = np.zeros(venv.num_envs)
    init_dones = np.zeros(venv.num_envs, dtype=bool)

    for env_idx, (act, rew, next_ob, done) in enumerate(zip(init_acts, init_rews, obs, init_dones)):
        step_dict = dict(acts=act, rews=rew, next_obs=next_ob, dones=done)
        trajectories_accum.add_step(env_idx, step_dict)


 
    active = np.ones(venv.num_envs, dtype=bool)
    dones = np.zeros(venv.num_envs, dtype=bool)

    last_sum = 4

    print("Start generate Traj")
    while np.any(active):
        acts, _ = policy.predict(obs,deterministic=True)
        next_obs, rews, dones, infos = venv.step(acts)

        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts[active],
            rews[active],
            next_obs[active],
            dones[active]
        )

        trajectories.extend(new_trajs)

        active &= ~dones
        #print(np.sum(active))
        if (np.sum(active) < last_sum):
            last_sum = np.sum(active)
            print("active env sum ", np.sum(active))
            print("active env", active)

    print("END")

    return trajectories
