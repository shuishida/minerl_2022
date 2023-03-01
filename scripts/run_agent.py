import random
from argparse import ArgumentParser

import aicrowd_gym
import minerl

import sys

import os
import torch
import numpy as np
from typing import List

sys.path.append(".")
from config import EVAL_EPISODES, EVAL_MAX_STEPS
from src.minerl.setup import setup_model
from src.minerl.agent import load_base_agent, MineSORLAgent
from const import get_env_name, ROOT_PATH
from src.utils.file_utils import load_yaml


def save_video_from_images(images: List[np.ndarray], save_path, fps=30):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import cv2
    import glob

    print(save_path)
    height, width, layers = images[0].shape
    size = (width, height)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(images)):
        out.write(images[i][:, :, ::-1])    # RGB to BGR
    out.release()


def eval_rollout(env_shorthand,
                 model_name,
                 agent_model="data/VPT-models/foundation-model-1x.model",
                 agent_weights="data/VPT-models/foundation-model-1x.weights",
                 n_episodes=EVAL_EPISODES,
                 max_steps=EVAL_MAX_STEPS,
                 show=False,
                 record_dir=None):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env_name = get_env_name(env_shorthand)

    env = aicrowd_gym.make(env_name)
    base_agent = load_base_agent(env, agent_model, agent_weights, DEVICE="cuda")
    config = load_yaml(os.path.join(ROOT_PATH, f"./config/{model_name}.yaml"))
    model = setup_model(base_agent, config, model_name)

    il_weights = os.path.join(ROOT_PATH, "train", config["agent"]["name"], f"{env_name}.pt")
    model.load_state_dict(torch.load(il_weights, map_location="cpu"))
    model.eval()
    model.to("cuda")
    agent = MineSORLAgent(base_agent, model.sample_actions)

    for i_episode in range(n_episodes):
        obs = env.reset()
        agent.reset()
        obsvs = [obs["pov"]]

        counter = 0
        fence_type = False
        active_hotbar = 1
        stage_started = False
        attack_repeat = 50
        continuous_attack = np.zeros(attack_repeat)
        attack_index = 0

        for i in range(int(max_steps)):
            action, pred_dones = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0   # int(pred_dones)

            for j in range(1, 10):
                if action[f"hotbar.{j}"]:
                    active_hotbar = j
            if stage_started:
                counter += 1

            if env_shorthand in ["cave", "waterfall", "animal"]:
                action["inventory"] = 0
            if env_shorthand == "cave":
                continuous_attack[attack_index] = int(action["forward"] == 0 and action["attack"] == 1)
                attack_index = (attack_index + 1) % attack_repeat
                if continuous_attack.mean() > 0.8:
                    action["jump"] = 1
                    action["camera"] = [0, 5]
                    action["back"] = 1
            if env_shorthand == "waterfall":
                action["hotbar.1"] = 1
                action['hotbar.2'] = 0
                action['hotbar.3'] = 0
                action['hotbar.4'] = 0
                action['hotbar.5'] = 0
                action['hotbar.6'] = 0
                action['hotbar.7'] = 0
                action['hotbar.8'] = 0
                action['hotbar.9'] = 0
                action["drop"] = 0
                if stage_started:
                    # action["use"] = 0
                    action["forward"] = 0
                    action["left"] = 0
                    action["right"] = 0
                    action['use'] = 0
                    if 0 < counter <= 36:
                        action['use'] = 1 if counter % 18 == 0 else 0
                        action["camera"] = [0, 5]
                        action["back"] = 0
                    if 36 <= counter:
                        action["back"] = 1
                    if 120 <= counter:
                        action["back"] = 0
                        action['use'] = 1 if counter % 18 == 0 else 0
                        action["jump"] = 0
                        action["attack"] = 0
                    if counter > 240:
                        action["ESC"] = 1
                else:
                    if i < 300:
                        action['use'] = 0
                    elif action["use"] == 1:
                        stage_started = True
            if env_shorthand == "animal":
                if action["use"] and active_hotbar >= 3:
                    stage_started = True
                if not stage_started:
                    action["inventory"] = 0
                if stage_started:
                    if random.random() < 0.1:
                        fence_type = not fence_type
                    action['hotbar.1'] = int(fence_type)
                    action['hotbar.2'] = int(not fence_type)
                    action['hotbar.3'] = 0
                    action['hotbar.4'] = 0
                    action['hotbar.5'] = 0
                    action['hotbar.6'] = 0
                    action['hotbar.7'] = 0
                    action['hotbar.8'] = 0
                    action['hotbar.9'] = 0
                    action['attack'] = action["attack"] if random.random() < 0.5 else 0
                    action["jump"] = action["jump"] if random.random() < 0.5 else 0
                    action['forward'] = action['forward'] if random.random() < 0.3 else 0
                    action["left"] = action['left'] if random.random() < 0.5 else 0
                    action["right"] = action['right'] if random.random() < 0.5 else 0
                    action["back"] = action['back'] if random.random() < 0.5 else 0
                    action["use"] = action["use"] if random.random() < 0.9 else 1
                if counter > 1500:
                    action['forward'] = 0
                    action["back"] = int(random.random() < 0.5)
                if counter > 1530:
                    action["ESC"] = 1
            if env_shorthand == "house":
                if action["use"] and active_hotbar >= 3:
                    stage_started = True
                if not stage_started:
                    action["inventory"] = 0
                else:
                    action['attack'] = action["attack"] if random.random() < 0.5 else 0
                    action["jump"] = action["jump"] if random.random() < 0.5 else 0
                    action['forward'] = action['forward'] if random.random() < 0.3 else 0
                    action["left"] = action['left'] if random.random() < 0.5 else 0
                    action["right"] = action['right'] if random.random() < 0.5 else 0
                    action["back"] = action['back'] if random.random() < 0.5 else 0
                    action["use"] = action["use"] if random.random() < 0.9 else 1
                if counter > 1500:
                    action['forward'] = 0
                    action["back"] = int(random.random() < 0.5)
                if counter > 1530:
                    action["ESC"] = 1

            obs, _, done, info = env.step(action)
            obsvs.append(obs["pov"])
            if show:
                env.render()
            if done:
                print("done")

                if record_dir:
                    save_video_from_images(obsvs, os.path.join(record_dir, f"run_{i_episode}.mp4"))

                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, default="data/VPT-models/foundation-model-1x.weights",
                        help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--base_model", type=str, default="data/VPT-models/foundation-model-1x.model",
                        help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default="cave")
    parser.add_argument('--model', help='model name', default='bc')
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--record", "-rec", type=str, help="Path to save recording.")

    args = parser.parse_args()

    eval_rollout(args.env, args.model, args.base_model, args.weights, n_episodes=100, max_steps=3600, show=args.show, record_dir=args.record)


"""
python run_agent.py --weights train/1x/MineRLBasaltFindCave.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltFindCave-v0 --show
python run_agent.py --weights train/1x/MineRLBasaltMakeWaterfall.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltMakeWaterfall-v0 --show
python run_agent.py --weights train/1x/MineRLBasaltCreateVillageAnimalPen.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltCreateVillageAnimalPen-v0 --show
python run_agent.py --weights train/1x/MineRLBasaltBuildVillageHouse.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltBuildVillageHouse-v0 --show
"""