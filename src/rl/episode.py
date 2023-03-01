import math
import random
import shutil
from copy import deepcopy
from multiprocessing import Manager, Value, Lock
from typing import Union, Any, Callable, Optional

import gym
import numpy as np
import torch
import os

from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate


from src.utils.file_utils import JSONIO
from src.utils.nested_utils import to_cpu
from src.utils.torch_utils import seed_worker


class Episode:
    def __init__(self, buffer: 'EpisodeBuffer', episode_id: Union[str, int], meta_data: Any = None):
        self.buffer = buffer
        self.id = episode_id
        self.index = int(episode_id)
        self.meta_data = meta_data
        self.obsvs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.options = []
        self.preds = []
        self.rgbs = []
        self.hiddens = []

    @property
    def firsts(self):
        n_obsvs = len(self.obsvs)
        if not n_obsvs: return []
        return [True] + [False] * (n_obsvs - 1)

    def __len__(self):
        return len(self.obsvs)

    def __getitem__(self, index):
        """
        Your typical RL replay buffer
        :param index:
        :return: (obsv, action, new_obsv, new_reward, new_done, info, pred)
        """
        if index >= len(self):
            raise IndexError
        return self.obsvs[index], self.actions[index], self.obsvs[index + 1], self.rewards[index + 1], self.dones[index + 1], \
               self.infos[index], self.preds[index]

    @classmethod
    def from_file(cls, buffer: 'EpisodeBuffer', filepath):
        data = torch.load(filepath, map_location="cpu")
        episode = Episode(buffer, data["id"], data.get("meta_data"))
        episode.obsvs = data["obsvs"]
        episode.actions = data["actions"]
        episode.rewards = data["rewards"]
        episode.dones = data["dones"]
        episode.infos = data["infos"]
        episode.preds = data["preds"]
        episode.rgbs = data["rgbs"]
        buffer.add_cache(episode)
        return episode

    @classmethod
    def from_id(cls, buffer: 'EpisodeBuffer', id):
        if id in buffer.cache:
            return buffer.cache[id]
        return cls.from_file(buffer, os.path.join(buffer.save_dir, f"{id}.pt"))

    @property
    def filepath(self):
        return self.buffer.get_episode_path(self.id)

    def add_inputs(self, obsv, reward, done, info):
        assert len(self.obsvs) == len(self.actions)

        self.obsvs.append(obsv)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def add_outputs(self, action, option=None, pred=None, rgb=None):
        self.actions.append(action)
        self.options.append(option)
        self.preds.append(pred)
        self.rgbs.append(rgb)
        assert len(self.obsvs) == len(self.actions)

    def get_all(self):
        return self.obsvs, self.actions, self.rewards, self.dones

    def get_fixed_length(self, index, length, obsv_transforms=None, action_transforms=None):
        """
        Intended to be used only for training on completed episodes.
        If there aren't enough samples to satisfy the sample length,
        the sample will be filled with repeated final observations, random actions, reward = 0 and done = True.

        :param index: starting timestep of sample t_start
        :param length: sample length delta_t
        :param obsv_transforms:
        :param action_transforms:
        :return: (obsvs, actions, rewards, firsts, dones), all from t_start to t_start + delta_t
        """
        if index >= len(self.obsvs) + length - 1:
            raise IndexError
        if index == -1:  # get most recent
            index = len(self.obsvs) - 1

        max_index = index + 1
        min_index = max(max_index - length, 0)
        fill_before = max(length - max_index, 0)

        def fill_fixed_length(seq: list, get_before: Callable[[list], Any], get_after: Optional[Callable[[list], Any]] = None):
            extract = [get_before(seq)] * fill_before + seq[min_index:max_index]
            fill_after = length - len(extract)
            if fill_after:
                extract += [get_after(seq)] * fill_after  # repeat default to fill length
            return extract

        def process(x: list, before: Callable[[list], Any], after: Callable[[list], Any] = None, transforms=None):
            x = fill_fixed_length(x, before, after)
            x = default_collate(x)
            if transforms:
                x = transforms(x)
            return x

        action_space = self.buffer.action_space

        def gen_rand_options(x):
            return int(self.buffer.n_options * random.random()) if self.buffer.n_options else 0

        return (
            process(self.obsvs, lambda x: x[0], lambda x: x[-1], transforms=obsv_transforms),  # obsvs
            process(self.actions, lambda x: action_space.sample(), lambda x: action_space.sample(), action_transforms),  # actions
            process(self.rewards, lambda x: 0.0, lambda x: 0.0).float(),  # rewards
            process(self.firsts, lambda x: True, lambda x: False),  # dones
            process(self.dones, lambda x: False, lambda x: True),  # dones
            process([o if o is not None else gen_rand_options(o) for o in self.options], gen_rand_options, gen_rand_options),  # options
        )

    def save(self):
        data = dict(
            meta_data=self.meta_data,
            id=self.id,
            obsvs=self.obsvs,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            infos=self.infos,
            options=self.options,
            preds=self.preds
        )
        data = to_cpu(data)
        torch.save(data, self.filepath)
        self.buffer.register_episode(self)
        return np.sum(self.rewards)


class EpisodeBuffer:
    lock = Lock()

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, n_options: int = None,
                 save_dir: str = None, clear: bool = False, max_episodes: Union[int, None] = None, sample_length: int = None,
                 cache_size: int = None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_options = n_options
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.sample_length = sample_length  # most common RL frameworks assume sample length of 2 (i.e. s_t, s_{t+1})

        if save_dir:
            if os.path.exists(save_dir):
                if clear:
                    shutil.rmtree(save_dir)
                    os.makedirs(save_dir)
                else:
                    print(f"Loading existing data at {save_dir}")
            else:
                os.makedirs(save_dir)

        manager = Manager()
        self.samples_info = manager.dict(JSONIO(os.path.join(save_dir, "samples.json") if save_dir else None))
        self._calc_sample_index()
        self.episode_ids = list(self.samples_info.keys())
        self._index = int(self.episode_ids[-1]) + 1 if len(self.samples_info) else 0

        self.cache_size = cache_size
        self.cache = {}

    def get_episode_path(self, episode_id: str):
        return os.path.join(self.save_dir, f"{episode_id}.pt") if self.save_dir else None

    def _calc_sample_index(self):
        sample_index = []
        for episode_id, length in self.samples_info.items():
            for index in range(length + self.sample_length - 2):
                sample_index.append((episode_id, index))
        self._sample_index = sample_index
        self._len = len(sample_index)

    def register_episode(self, episode: Episode):
        self.samples_info[episode.id] = len(episode)
        self._calc_sample_index()
        self.add_cache(episode)

    def delete_episode(self, episode_id: str):
        if episode_id in self.samples_info:
            del self.samples_info[episode_id]
        self._calc_sample_index()
        filepath = self.get_episode_path(episode_id)
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

    @property
    def n_episodes(self):
        return len(self.samples_info)

    def __len__(self):
        return self._len

    def add_cache(self, episode: Episode):
        if not self.cache_size: return  # if cash_size is 0 or no cash_size defined, do not add to cash
        while len(self.cache) >= self.cache_size:  # discard old cash
            self.cache.pop(next(iter(self.cache.keys())))
        self.cache[episode.id] = episode

    def get(self, index, obsv_transforms, action_transforms):
        episode_id, index = self._sample_index[index]
        episode = Episode.from_id(self, episode_id)
        return episode.get_fixed_length(index, self.sample_length, obsv_transforms, action_transforms)

    def create_episode(self, meta_data=None) -> Episode:
        with self.lock:
            episode_id = str(self._index)
            episode = Episode(self, episode_id=episode_id, meta_data=meta_data)
            self.episode_ids.append(episode_id)
            self._index += 1

            if self.max_episodes and len(self.episode_ids) >= self.max_episodes:
                assert len(self.episode_ids) == self.max_episodes
                remove_episode_id = self.episode_ids.pop(0)
                self.delete_episode(remove_episode_id)
        return episode

    def get_episode(self, i_episode) -> Episode:
        episode_ids = list(self.samples_info.keys())
        assert 0 <= i_episode < len(episode_ids), "episode index out of range"
        episode_id = episode_ids[i_episode]
        return Episode.from_id(self, episode_id)


class FixedEpisodeDataset(Dataset):
    def __init__(self, buffer: EpisodeBuffer, obsv_transforms=None, action_transforms=None):
        self.buffer = buffer
        self.samples_info = deepcopy(dict(self.buffer.samples_info))
        self.obsv_transforms = obsv_transforms
        self.action_transforms = action_transforms

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        self.buffer.get(index, self.obsv_transforms, self.action_transforms)


class IterEpisodeDataset(IterableDataset):
    def __init__(self, buffer: EpisodeBuffer, max_steps: int = None, obsv_transforms=None, action_transforms=None):
        seed_worker()
        self.buffer = buffer
        self.max_steps = max_steps
        self.count = 0
        self.obsv_transforms = obsv_transforms
        self.action_transforms = action_transforms

    def __iter__(self):
        if not len(self.buffer): return
        worker_info = torch.utils.data.get_worker_info()
        if self.max_steps is None:
            steps_per_worker = np.inf
        elif worker_info is None:
            steps_per_worker = self.max_steps
        else:
            steps_per_worker = int(math.ceil(self.max_steps / float(worker_info.num_workers)))
        while self.count < steps_per_worker:
            n = len(self.buffer)
            i = int(n * random.random())  # faster than np.random.randint(n)
            yield self.buffer.get(i, self.obsv_transforms, self.action_transforms)
            self.count += 1
