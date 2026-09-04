import random
import time
from temporalio import activity
from temporalio.exceptions import ApplicationError


# Global attempt counter to simulate flaky behavior
_attempt_counters: dict[str, int] = {}


@activity.defn
async def flaky_data_fetch(endpoint: str) -> dict:
    """
    Simulates a flaky external API call.
    Fails the first 2 attempts, succeeds on the 3rd.
    Temporal will automatically retry it!
    """
    key = f"flaky_data_fetch_{endpoint}"
    _attempt_counters[key] = _attempt_counters.get(key, 0) + 1
    attempt = _attempt_counters[key]

    activity.logger.info(f"Attempt #{attempt} to fetch from: {endpoint}")

    if attempt <= 2:
        raise RuntimeError(f"Network timeout on attempt #{attempt} — Temporal will retry!")

    time.sleep(0.3)
    return {"endpoint": endpoint, "data": {"value": 42, "status": "success"}, "attempt": attempt}


@activity.defn
async def validate_data(data: dict) -> dict:
    """
    Validates incoming data.
    Raises a NON-RETRYABLE ApplicationError if data is invalid.
    Temporal will NOT retry non-retryable errors.
    """
    activity.logger.info(f"Validating data: {data}")

    if not data.get("data"):
        # Non-retryable: bad data won't get better with retries
        raise ApplicationError(
            "Data validation failed: 'data' field is missing or empty",
            non_retryable=True,
        )

    value = data["data"].get("value")
    if value is None or value < 0:
        raise ApplicationError(
            f"Invalid value: {value}. Must be >= 0",
            non_retryable=True,
        )

    return {"valid": True, "value": value, "source": data.get("endpoint")}


@activity.defn
async def process_with_heartbeat(items: list[str]) -> list[str]:
    """
    Processes a list of items and sends heartbeats.
    Heartbeats tell Temporal 'I'm still alive' for long-running activities.
    If the heartbeat_timeout is exceeded, Temporal will retry the activity.
    """
    results = []
    total = len(items)

    for i, item in enumerate(items):
        # Heartbeat: report progress and check for cancellation
        activity.heartbeat(f"Processing item {i+1}/{total}: {item}")

        # Check if activity was cancelled (heartbeat raises CancelledError if so)
        time.sleep(0.2)  # Simulate work
        results.append(f"processed_{item}")

    return results


@activity.defn
async def compensate_cleanup(resource_id: str) -> str:
    """
    Compensation activity — runs when the workflow needs to undo work.
    Always designed to be idempotent (safe to call multiple times).
    """
    activity.logger.info(f"Running compensation/cleanup for resource: {resource_id}")
    time.sleep(0.1)
    return f"Cleaned up resource: {resource_id}"
