"""
Logging Configuration
Bilingual Document Q&A (RAG) System

Provides structured logging without leaking sensitive keys or full user queries.
"""

import logging
import sys

def setup_logger(name: str = "bilingual_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [req_id=%(request_id)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
