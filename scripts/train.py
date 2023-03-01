# Train one model for each task
import os

import sys

sys.path.append(".")

from scripts.process_dataset import precompute_vpt_embedding_for_task
from src.minerl.setup import RLSetup
from const import TASK_CAVE, TASK_WATERFALL, TASK_ANIMAL, TASK_HOUSE


def main():
    print("=== Extracting FindCave features ===")
    precompute_vpt_embedding_for_task(TASK_CAVE, overwrite=False)

    print("=== Extracting Waterfall features ===")
    precompute_vpt_embedding_for_task(TASK_WATERFALL, overwrite=False)

    print("=== Extracting AnimalPen features ===")
    precompute_vpt_embedding_for_task(TASK_ANIMAL, overwrite=False)

    print("=== Extracting VillageHouse features ===")
    precompute_vpt_embedding_for_task(TASK_HOUSE, overwrite=False)

    print("===Training FindCave model===")
    RLSetup.run({"env": "cave", "model": "bc", "epochs": 1})

    print("===Training Waterfall model===")
    RLSetup.run({"env": "waterfall", "model": "bc", "epochs": 1})

    print("===Training AnimalPen model===")
    RLSetup.run({"env": "animal", "model": "bc", "epochs": 1})

    print("===Training VillageHouse model===")
    RLSetup.run({"env": "house", "model": "bc", "epochs": 1})


if __name__ == "__main__":
    main()
