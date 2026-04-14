"""
Simple file logger for DocuBot Article Analyser.
Logs queries, retrieved snippets, answers, and errors to docubot.log.
"""

import logging
import os

_log_path = os.path.join(os.path.dirname(__file__), "docubot.log")

logging.basicConfig(
    filename=_log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger("docubot")


def log_query(mode: str, query: str) -> None:
    _logger.info("QUERY [%s]: %s", mode, query)


def log_snippets(snippets: list) -> None:
    _logger.info("RETRIEVED %d snippet(s):", len(snippets))
    for filename, text, score in snippets:
        preview = text[:120].replace("\n", " ")
        _logger.info("  [%s] score=%s — %s...", filename, score, preview)


def log_answer(answer: str) -> None:
    preview = answer[:300].replace("\n", " ")
    _logger.info("ANSWER: %s...", preview)


def log_themes(themes: list) -> None:
    _logger.info("THEMES: %s", ", ".join(themes))


def log_argument_analysis(claim: str) -> None:
    _logger.info("ARGUMENT ANALYSIS for claim: %s", claim)


def log_error(context: str, error: Exception) -> None:
    _logger.error("ERROR in %s: %s", context, error)
