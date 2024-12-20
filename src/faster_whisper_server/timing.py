import time
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@contextmanager
def timing(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name} took {end - start:.3f} seconds")
