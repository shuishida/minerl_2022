import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get('MINERL_DATA_ROOT', os.path.join(ROOT_PATH, "data"))
VPT_PATH = os.path.join(DATA_PATH, "VPT-models")

TASK_DIAMOND = "MineRLObtainDiamondShovel-v0"
TASK_CAVE = "MineRLBasaltFindCave-v0"
TASK_WATERFALL = "MineRLBasaltMakeWaterfall-v0"
TASK_ANIMAL = "MineRLBasaltCreateVillageAnimalPen-v0"
TASK_HOUSE = "MineRLBasaltBuildVillageHouse-v0"

TASKS = [TASK_CAVE, TASK_WATERFALL, TASK_ANIMAL, TASK_HOUSE]

n_demo_per_task = {task: len(os.listdir(os.path.join(DATA_PATH, task))) // 2 for task in TASKS}

USING_FULL_DATASET = min(n_demo_per_task.values()) > 1000


def get_env_name(shorthand):
    if shorthand == "cave":
        return TASK_CAVE
    elif shorthand == "animal":
        return TASK_ANIMAL
    elif shorthand == "house":
        return TASK_HOUSE
    elif shorthand == "waterfall":
        return TASK_WATERFALL
    else:
        raise ValueError("Unknown shorthand")
