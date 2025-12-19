import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def is_retryable_httpx_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.RequestError) and not isinstance(
        exc, httpx.HTTPStatusError
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def retryable_request(
    logger: logging.Logger,
    attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 8.0,
):
    return retry(
        retry=retry_if_exception(is_retryable_httpx_error),
        wait=wait_random_exponential(min=min_wait, max=max_wait),
        stop=stop_after_attempt(attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
