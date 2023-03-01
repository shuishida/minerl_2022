from argparse import ArgumentParser
import pickle

import aicrowd_gym
import minerl

import sys
sys.path.append(".")

from openai_vpt.agent import MineRLAgent


def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=False):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for _ in range(n_episodes):
        obs = env.reset()
        agent.reset()

        for _ in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            if done:
                print("done")
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, default="data/VPT-models/rl-from-early-game-2x.weights",
                        help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="data/VPT-models/2x.model",
                        help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default="MineRLObtainDiamondShovel-v0")
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)


"""
python run_agent.py --weights train/1x/MineRLBasaltFindCave.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltFindCave-v0 --show
python run_agent.py --weights train/1x/MineRLBasaltMakeWaterfall.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltMakeWaterfall-v0 --show
python run_agent.py --weights train/1x/MineRLBasaltCreateVillageAnimalPen.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltCreateVillageAnimalPen-v0 --show
python run_agent.py --weights train/1x/MineRLBasaltBuildVillageHouse.weights --model data/VPT-models/foundation-model-1x.model --env MineRLBasaltBuildVillageHouse-v0 --show
"""