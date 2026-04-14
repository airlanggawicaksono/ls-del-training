import argparse

from config_types_ee import EETrainConfig
from trainer_utils import _parse_config_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Early-Exit Tuning for LLaMA")
    parser.add_argument("--config", required=True, help="Path to key=value config file")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    raw = _parse_config_file(args.config)
    config = EETrainConfig.model_validate(raw)

    from ee.train import run_ee_training

    run_ee_training(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
