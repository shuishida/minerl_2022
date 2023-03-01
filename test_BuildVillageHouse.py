from scripts.run_agent import eval_rollout


def main():
    print("=== Evaluating VillageHouse ===")
    eval_rollout("house", "bc")


if __name__ == "__main__":
    main()
