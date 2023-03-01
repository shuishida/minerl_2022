from scripts.run_agent import eval_rollout


def main():
    print("=== Evaluating AnimalPen ===")
    eval_rollout("animal", "bc")


if __name__ == "__main__":
    main()
