import warnings
import os
import tensorflow as tf
import absl.logging

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


def main():
    disable_warnings()

    logic = CommunicationLogic()
    logic.start()


if __name__ == '__main__':
    main()

