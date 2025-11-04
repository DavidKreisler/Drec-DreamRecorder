import warnings
import os
import tensorflow as tf
import absl.logging
import argparse

from scripts.Logic.communicationLogic import CommunicationLogic


def disable_warnings():
    # Suppress only TensorFlow deprecation warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERROR
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Suppress Python warnings (e.g., UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*is deprecated and will be removed.*")
    absl.logging.set_verbosity(absl.logging.ERROR)

def get_arguments():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--mode", "-m",
                        choices=["cli", "server"],
                        default="cli",
                        help="Run mode: 'cli' (default) or 'server'")

    args = parser.parse_args()
    return args

def main():
    disable_warnings()

    args = get_arguments()

    logic = CommunicationLogic(mode=args.mode)
    logic.start()

    # ToDo: add endpoints to get current state


if __name__ == '__main__':
    main()

