import json
from typing import Any, Callable, Optional

import gym
import torch
import os

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class EpisodeDataset(Dataset):
    def __init__(self, dataset_dir: str, action_space: gym.Space, action_transforms, sample_length: int) -> None:
        super().__init__()
        self.action_space = action_space
        self.action_transforms = action_transforms
        self.sample_length = sample_length  # most common RL frameworks assume sample length of 2 (i.e. s_t, s_{t+1})
        self.data_dir = dataset_dir
        with open(os.path.join(dataset_dir, "counts.json"), "r") as f:
            self.counts = json.load(f)
        self.index_to_episode = {}
        count_sum = 0
        for episode_id, length in self.counts.items():
            if os.path.exists(os.path.abspath(os.path.join(self.data_dir, episode_id + ".pt"))) \
                    and os.path.exists(os.path.abspath(os.path.join(self.data_dir, episode_id + "_actions.pt"))):
                for i in range(length + self.sample_length - 2):
                    self.index_to_episode[count_sum] = (episode_id, length, i)
                    count_sum += 1
        self._len = count_sum

    def _load_episode_files(self, episode_id):
        vpt_emb_path = os.path.abspath(os.path.join(self.data_dir, episode_id + ".pt"))
        vpt_emb = torch.load(vpt_emb_path)
        vpt_action_path = os.path.abspath(os.path.join(self.data_dir, episode_id + "_actions.pt"))
        vpt_action = torch.load(vpt_action_path)
        return vpt_emb, vpt_action

    def get_fixed_length(self, index):
        episode_id, episode_len, index = self.index_to_episode[index]
        vpt_emb, vpt_action = self._load_episode_files(episode_id)

        length = self.sample_length

        if index >= episode_len + length - 1:
            raise IndexError

        max_index = index + 1
        min_index = max(max_index - length, 0)
        fill_before = max(length - max_index, 0)

        def fill_fixed_length(seq: list, get_before: Callable[[list], Any], get_after: Optional[Callable[[list], Any]] = None, transforms=None):
            x = seq[min_index:max_index]
            if transforms:
                x = [transforms(e) for e in x]
            extract = [get_before(seq)] * fill_before + list(x)
            fill_after = length - len(extract)
            if fill_after:
                extract += [get_after(seq)] * fill_after  # repeat default to fill length
            return extract

        def process(x: list, before: Callable[[list], Any], after: Callable[[list], Any] = None, transforms=None):
            x = fill_fixed_length(x, before, after, transforms)
            x = default_collate(x)
            return x

        action_space = self.action_space
        action_sampler = lambda x: action_space.sample()

        return (
            process(vpt_emb, lambda x: x[0], lambda x: x[-1]),  # obsvs
            process(vpt_action, action_sampler, action_sampler,
                    lambda x: {k: v.squeeze(0).squeeze(0) for k, v in self.action_transforms(x).items()}),  # actions
            torch.zeros(length, dtype=torch.float),  # rewards
            process([True] + [False] * (episode_len - 1), lambda x: True, lambda x: False),  # firsts
            process([False] * (episode_len - 1) + [True], lambda x: False, lambda x: True),  # dones
        )

    def __getitem__(self, index):
        return self.get_fixed_length(index)

    def __len__(self):
        return self._len
