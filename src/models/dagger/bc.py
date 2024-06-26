"""Behavioural Cloning (BC).

Trains policy by applying supervised learning to a fixed dataset of (observation,
action) pairs generated by some expert demonstrator.
"""

import dataclasses

import gymnasium as gym
import numpy as np

from tqdm import tqdm
from stable_baselines3.common import policies, torch_layers, utils, vec_env


import torch
from torch.utils.data import Dataset, DataLoader

from .networks import FeedForward32Policy    


@dataclasses.dataclass(frozen=True)
class BCTrainingMetrics:
    """Container for the different components of behavior cloning loss."""
    neglogp: torch.Tensor
    entropy: torch.Tensor
    ent_loss: torch.Tensor  # set to 0 if entropy is None
    prob_true_act: torch.Tensor
    l2_norm: torch.Tensor
    l2_loss: torch.Tensor
    loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class BehaviorCloningLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        obs: torch.Tensor,
        acts: torch.Tensor,
    ) -> BCTrainingMetrics:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        # policy.evaluate_actions's type signatures are incorrect.
        # See https://github.com/DLR-RM/stable-baselines3/issues/1679
        (_, log_prob, entropy) = policy.evaluate_actions(obs, acts)

        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * (entropy if entropy is not None else torch.zeros(1))
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return BCTrainingMetrics(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            loss=loss,
        )




class BC:
    """Behavioral cloning (BC).

    Recovers a policy via supervised learning from observation-action pairs.
    """
    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        policy: policies.ActorCriticPolicy = None,
        demonstrations: DataLoader = None,
        batch_size: int = 32,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: torch.device = "cpu",
        
    ):
        """Builds BC.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            rng: the random state to use for the random number generator.
            policy: a Stable Baselines3 policy; if unspecified,
                defaults to `FeedForward32Policy`.
            demonstrations: dataLoader 
            batch_size: The number of samples in each batch of expert data.
            minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until `batch_size` examples
                are processed before making an optimization step. This
                is useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `batch_size`.
                Optional, defaults to `batch_size`.
            optimizer_cls: supervised training optimizer
            optimizer_kwargs: keyword arguments, excluding learning rate and
                weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead), or if the batch size is not a multiple
                of the minibatch size.
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.demonstrations = demonstrations
        
        self.action_space = action_space
        self.observation_space = observation_space

        self.rng = rng
        print("observation_space", observation_space)
        if policy is None:
            extractor = (
                torch_layers.CombinedExtractor
                if isinstance(observation_space, gym.spaces.Dict)
                else 
                torch_layers.FlattenExtractor
            )
            policy = FeedForward32Policy(
                observation_space=observation_space,
                action_space=action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: torch.finfo(torch.float32).max,
                features_extractor_class=extractor,
            )
        
        self._policy = policy.to(utils.get_device(device))

        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        optimizer_kwargs = optimizer_kwargs or {}

        self.optimizer = optimizer_cls(self.policy.parameters(), **optimizer_kwargs)
        self.loss_calculator = BehaviorCloningLossCalculator(ent_weight, l2_weight)

    def set_demonstrations(self, demonstrations: DataLoader) -> None:
            self.demonstrations =demonstrations 


    def train(
        self,
        *,
        n_epochs: int = None,
        n_batches: int = None,
        log_interval: int = 500,
        log_rollouts_venv: vec_env.VecEnv = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`. Note, that when you specify
        `n_batches` smaller than the number of batches in an epoch, the `on_epoch_end`
        callback will never be called.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode lengtorch. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_tensorboard: If True, then start plotting to Tensorboard from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """

        assert self.demonstrations is not None

        for epoch in range(n_epochs):
            print("Epoch ", epoch)
            for batch_count, batch in enumerate(tqdm(self.demonstrations, desc='Training '), 0):

                obs, acts = batch 
                obs  = obs.to(self.device)
                acts = acts.to(self.device)

                training_metrics = self.loss_calculator(self.policy, obs, acts)

                loss = training_metrics.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_count % 100 == 0:
                    print("loss", loss)
                    

           


    @property
    def policy(self) -> policies.ActorCriticPolicy:
        return self._policy

