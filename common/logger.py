import logging

logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%m/%d %I:%M:%S")

logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)