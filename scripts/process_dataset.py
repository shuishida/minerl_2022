# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.

from argparse import ArgumentParser
import json
import pickle
import os
import time
from collections import defaultdict
from multiprocessing import Process

import matplotlib.pyplot as plt
import gym
import minerl
import torch as th
import numpy as np

import sys

from typing import List

sys.path.append(".")

from src.utils.model_utils import load_model_parameters
from const import USING_FULL_DATASET, DATA_PATH, VPT_PATH, TASK_CAVE, TASK_WATERFALL, TASK_ANIMAL, TASK_HOUSE
from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader, FullSweepDataLoader
from openai_vpt.lib.tree_util import tree_map

# Needs to be <= number of videos
BATCH_SIZE = 16
DEVICE = "cuda"

LOSS_REPORT_RATE = 100


def precompute_vpt_embeddings(data_dir, save_dir, in_model, in_weights, task_name, overwrite=True, time_limit: int = 18 * 60 * 60):
    if os.path.exists(save_dir):
        if overwrite:
            print(f"Overwriting {save_dir}")
        else:
            print(f"Skipping {save_dir}")
            return
    else:
        os.makedirs(save_dir)

    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    env.close()

    policy = agent.policy

    data_loader = FullSweepDataLoader(
        dataset_dir=data_dir,
        batch_size=BATCH_SIZE
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    episode_embeddings = defaultdict(list)
    episode_actions = defaultdict(list)
    episode_step_count_tmp = defaultdict(int)
    episode_step_count = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    for batch_i, batch in enumerate(data_loader):
        for image, action, episode_id, timestep, done in zip(*batch):
    
            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            with th.no_grad():
                pi_distribution, _, new_agent_state, embedding = policy.take_step(
                    agent_obs,
                    dummy_first,
                    agent_state
                )

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state
            episode_step_count_tmp[episode_id] += 1
            episode_embeddings[episode_id].append(embedding.detach().cpu())
            episode_actions[episode_id].append(action)

            if done:
                # A work-item was done. Remove hidden state
                th.save(th.cat(episode_embeddings[episode_id]), os.path.join(save_dir, f"{episode_id}.pt"))
                th.save(episode_actions[episode_id], os.path.join(save_dir, f"{episode_id}_actions.pt"))
                # print(episode_id, "deleted", episode_step_count[episode_id], len(episode_embeddings[episode_id]))
                del episode_hidden_states[episode_id]
                del episode_embeddings[episode_id]
                del episode_actions[episode_id]
                episode_step_count[episode_id] = episode_step_count_tmp[episode_id]

        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Task: {task_name}, Time: {time_since_start:.2f}, Batches: {batch_i}.", len(episode_step_count))
            # print(episode_step_count)
            if time_since_start > time_limit:
                print("Time limit reached, stopping")
                break

    with open(os.path.join(save_dir, "counts.json"), "w") as f:
        json.dump(episode_step_count, f)


def precompute_vpt_embedding_for_task(task, overwrite=True):
    precompute_vpt_embeddings(
        data_dir=os.path.join(DATA_PATH, task),
        save_dir=f"train/preprocess/{task}",
        in_model=os.path.join(VPT_PATH, "foundation-model-1x.model"),
        in_weights=os.path.join(VPT_PATH, "foundation-model-1x.weights"),
        task_name=task,
        overwrite=overwrite
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    args = parser.parse_args()

    print(f"=== Precomputing embeddings ===")
    if args.task == "cave":
        precompute_vpt_embedding_for_task(TASK_CAVE)
    elif args.task == "waterfall":
        precompute_vpt_embedding_for_task(TASK_WATERFALL)
    elif args.task == "animal":
        precompute_vpt_embedding_for_task(TASK_ANIMAL)
    elif args.task == "house":
        precompute_vpt_embedding_for_task(TASK_HOUSE)
    elif args.task == "all":
        for task in [TASK_CAVE, TASK_WATERFALL, TASK_ANIMAL, TASK_HOUSE]:
            precompute_vpt_embedding_for_task(task)
    else:
        raise NotImplementedError
