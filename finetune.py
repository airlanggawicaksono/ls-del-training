import argparse

from trainer_utils import load_train_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaMA decoder-only Trainer (config-file based).")
    parser.add_argument("--config", required=True, help="Path to key=value config file")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    config = load_train_config(args.config)
    run_training(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
