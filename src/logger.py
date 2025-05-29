"""
Logger factory module for the evaluation program.
"""

import logging


def create_logger() -> logging.Logger:
    """
    Creates the logger for the Jace application.
    """

    logger = logging.Logger("hle-eval-ollama")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s: %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
