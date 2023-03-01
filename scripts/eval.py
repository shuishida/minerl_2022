import sys

sys.path.append(".")
from scripts.run_agent import eval_rollout


def main():
    print("=== Evaluating FindCave ===")
    eval_rollout("cave", "bc")

    print("=== Evaluating Waterfall ===")
    eval_rollout("waterfall", "bc")

    print("=== Evaluating AnimalPen ===")
    eval_rollout("animal", "bc")

    print("=== Evaluating VillageHouse ===")
    eval_rollout("house", "bc")


if __name__ == "__main__":
    main()
