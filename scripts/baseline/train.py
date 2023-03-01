# Train one model for each task
import os

import sys
sys.path.append(".")

from scripts.baseline.behavioural_cloning import behavioural_cloning_train
from const import DATA_PATH, VPT_PATH


def main():
    print("===Training FindCave model===")
    behavioural_cloning_train(
        data_dir=os.path.join(DATA_PATH, "MineRLBasaltFindCave-v0"),
        in_model=os.path.join(VPT_PATH, "foundation-model-1x.model"),
        in_weights=os.path.join(VPT_PATH, "foundation-model-1x.weights"),
        out_weights="train/MineRLBasaltFindCave.weights"
    )

    print("===Training MakeWaterfall model===")
    behavioural_cloning_train(
        data_dir=os.path.join(DATA_PATH, "MineRLBasaltMakeWaterfall-v0"),
        in_model=os.path.join(VPT_PATH, "foundation-model-1x.model"),
        in_weights=os.path.join(VPT_PATH, "foundation-model-1x.weights"),
        out_weights="train/MineRLBasaltMakeWaterfall.weights"
    )

    print("===Training CreateVillageAnimalPen model===")
    behavioural_cloning_train(
        data_dir=os.path.join(DATA_PATH, "MineRLBasaltCreateVillageAnimalPen-v0"),
        in_model=os.path.join(VPT_PATH, "foundation-model-1x.model"),
        in_weights=os.path.join(VPT_PATH, "foundation-model-1x.weights"),
        out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    )

    print("===Training BuildVillageHouse model===")
    behavioural_cloning_train(
        data_dir=os.path.join(DATA_PATH, "MineRLBasaltBuildVillageHouse-v0"),
        in_model=os.path.join(VPT_PATH, "foundation-model-1x.model"),
        in_weights=os.path.join(VPT_PATH, "foundation-model-1x.weights"),
        out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    )


if __name__ == "__main__":
    main()
