from scripts.run_agent import eval_rollout


def main():
    print("=== Evaluating FindCave ===")
    eval_rollout("cave", "bc")


if __name__ == "__main__":
    main()
