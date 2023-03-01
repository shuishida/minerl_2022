import pickle

import torch
import torch.nn.functional as F
from openai_vpt.agent import MineRLAgent

DEVICE = "cuda"


class MineSORLAgent:
    def __init__(self, base_agent: MineRLAgent, sample_action):
        self.agent = base_agent
        self.sample_action = sample_action

    def reset(self):
        self.agent.reset()

    def get_action(self, minerl_obs):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self.agent._env_obs_to_agent(minerl_obs)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        with torch.no_grad():
            _, _, self.agent.hidden_state, state = self.agent.policy.take_step(
                agent_input, self.agent._dummy_first, self.agent.hidden_state)

            agent_action, done = self.sample_action(state)

        minerl_action = self.agent._agent_action_to_env(agent_action)

        return minerl_action, done


def load_base_agent(env, model_path, weights_path, DEVICE="cuda"):
    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    return agent
