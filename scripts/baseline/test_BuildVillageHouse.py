import os

import sys
sys.path.append(".")

from const import VPT_PATH
from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS


def main(run_local=False):
    run_agent_main(
        model=os.path.join(VPT_PATH, "foundation-model-1x.model"),
        weights="train/MineRLBasaltBuildVillageHouse.weights",
        env="MineRLBasaltBuildVillageHouse-v0",
        n_episodes=10 if run_local else EVAL_EPISODES,
        max_steps=int(1e4) if run_local else EVAL_MAX_STEPS,
        show=run_local
    )


if __name__ == "__main__":
    main(True)
