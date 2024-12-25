import logging
import os

logger = logging.getLogger(__name__)

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger.setLevel(log_level)

handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
