import logging
import os


def configure_logger():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/banking_assistant.log"),
            logging.StreamHandler(),
        ],
    )
