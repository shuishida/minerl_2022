import os
from typing import List, Any, Callable, Tuple, Optional
import sys

import gym
import torch
import numpy as np

import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from src.rl.env import VecEnv
from src.rl.episode import EpisodeBuffer, Episode
from src.utils.image_utils import save_video_from_images
from src.utils.model_utils import NoGrad
from src.utils.nested_utils import to_device, nested_rearrange, to_numpy


class Agent:
    def __init__(self, buffer: EpisodeBuffer, observation_space: gym.Space, action_space: gym.Space):
        self.buffer = buffer
        self.observation_space = observation_space
        self.action_space = action_space

        self.count_train_steps = 0

        self.active_episodes: List[Episode] = []
        self.actions = []

    def get_actions(self, episodes: List[Episode], is_train: bool = False, save_pred: bool = False):
        """
        :return: actions, outputs
        """
        raise NotImplementedError

    def act(self, episode: Episode, is_train: bool = False, save_pred: bool = False):
        actions, options, preds = self.get_actions([episode], is_train, save_pred)
        return actions[0], options[0], preds[0]

    def rollouts(self, env: VecEnv, buffer: EpisodeBuffer = None, n_steps: int = None, n_episodes: int = None,
                 is_train: bool = True, save_pred=False, render: bool = False, record_dir: Optional[str] = None, verbose=True):
        assert n_steps or n_episodes, "either n_steps or n_episodes have to be defined"
        if is_train:
            assert buffer is None
            buffer = self.buffer

        if is_train and self.active_episodes:
            # resume from last step
            active_episodes = list(self.active_episodes)
            actions = list(self.actions)
        else:
            active_episodes: List[Episode] = [None] * env.size
            actions = [None] * env.size

        count_steps = count_episodes = total_rewards = avg_returns = 0
        while (n_steps and count_steps < n_steps) or (n_episodes and count_episodes < n_episodes):

            # take step in vectorized environment
            meta_arr, resets, obsvs, rewards, dones, infos = env.step(actions)

            for i, (episode, meta_data, reset, *data) in enumerate(
                    zip(active_episodes, meta_arr, resets, obsvs, rewards, dones, infos)):
                # create a new episode if the previous episode has terminated
                if reset:
                    active_episodes[i] = episode = buffer.create_episode(meta_data)
                # add env inputs to episode
                episode.add_inputs(*data)

            with torch.no_grad():
                # agent makes decision
                actions, options, preds = self.get_actions(active_episodes, is_train=is_train, save_pred=save_pred)

            for i, (episode, action, option, pred) in enumerate(zip(active_episodes, actions, options, preds)):
                # add agent output to episode
                episode.add_outputs(action, option, pred if save_pred else None, env.get_rgb(i) if save_pred else None)

            if verbose:
                if n_steps:
                    sys.stdout.write(f"\r--- Rolling out {count_steps} / {n_steps} steps. Avg returns: {avg_returns}.")
                else:
                    sys.stdout.write(
                        f"\r--- Rolling out {count_episodes} / {n_episodes} episodes, {count_steps} steps. Avg returns: {avg_returns}")
                sys.stdout.flush()

            for episode, done, info in zip(active_episodes, dones, infos):
                if done and (is_train or n_steps or episode.index < n_episodes):
                    if record_dir:
                        save_video_from_images(episode.rgbs, os.path.join(record_dir, f"run_{count_episodes}.mp4"))

                    count_episodes += 1
                    total_rewards += episode.save()
                    avg_returns = total_rewards / count_episodes

            count_steps += len(meta_arr)
            if is_train:
                self.count_train_steps += len(meta_arr)

            if render:
                env.render()

        if verbose:
            sys.stdout.write("\r")
            sys.stdout.flush()

        if is_train:
            self.active_episodes = active_episodes
            self.actions = actions
        else:
            for episode in active_episodes:
                if episode.index >= count_episodes:
                    buffer.delete_episode(episode.id)

        return {
            "avg_returns": avg_returns,
            "count_episodes": count_episodes,
            "count_steps": count_steps,
        }


class RandomAgent(Agent):
    def get_actions(self, episode: List[Episode], is_train: bool = False, save_pred: bool = False):
        actions = [self.action_space.sample() for _ in range(len(episode))]
        return actions, [None] * len(actions), [None] * len(actions)


class GenericAgent(Agent):
    def __init__(
            self,
            model: nn.Module,
            buffer: EpisodeBuffer,
            observation_space: gym.Space,
            action_space: gym.Space,
            sample_actions: Callable,
            observation_transform=None,
            action_transform=None,
            epsilon_train: float = 0.0,
            epsilon_eval: float = 0.0,
    ):
        super().__init__(buffer, observation_space, action_space)
        self.model = model
        self.obsv_tf = observation_transform
        self.action_tf = action_transform
        self.sample_actions = sample_actions
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval or 0

    def get_actions(self, episodes: List[Episode], is_train: bool = False, save_pred: bool = False):
        batch_size = len(episodes)
        epsilon = self.epsilon_train if is_train else self.epsilon_eval
        if np.random.random() < epsilon:
            return [self.action_space.sample() for _ in range(len(episodes))], [None] * batch_size, [None] * batch_size
        with NoGrad(self.model):
            batch = default_collate([episode.get_fixed_length(-1, self.buffer.sample_length, self.obsv_tf, self.action_tf) for episode in episodes])
            batch = to_device(batch, next(self.model.parameters()).device)
            batch = nested_rearrange(batch, "b l ... -> l b ...")
            actions, options, preds = to_numpy(self.sample_actions(*batch, deterministic=not is_train))
        return actions, options, preds
