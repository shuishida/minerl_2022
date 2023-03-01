from scripts.run_agent import eval_rollout


def main():
    print("=== Evaluating Waterfall ===")
    eval_rollout("waterfall", "bc")


if __name__ == "__main__":
    main()
