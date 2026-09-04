from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from activities import (
        flaky_data_fetch,
        validate_data,
        process_with_heartbeat,
        compensate_cleanup,
    )


@workflow.defn
class RetryAndErrorWorkflow:
    """
    Project 04: Retry & Error Handling

    Demonstrates:
    - RetryPolicy: initial_interval, backoff_coefficient, max_attempts
    - Non-retryable errors (ApplicationError with non_retryable=True)
    - Heartbeat pattern for long-running activities
    - Catching ActivityError in the workflow
    - Compensation/cleanup on failure
    - Different retry policies for different activity types

    Key Insight:
    Temporal retries happen AUTOMATICALLY. You define the policy and
    Temporal handles the retry scheduling, backoff, and attempt tracking.
    You can see all retry attempts in the Web UI under each activity.
    """

    @workflow.run
    async def run(self, endpoint: str) -> str:
        workflow.logger.info(f"RetryAndError Workflow started for endpoint: {endpoint}")
        allocated_resource = None

        try:
            # ── Step 1: Flaky fetch with aggressive retry ──────────────────────
            # This activity will fail 2 times, then succeed.
            # Watch the retry attempts in the Web UI!
            workflow.logger.info("Step 1: Fetching data (will retry on failure)...")
            raw_data = await workflow.execute_activity(
                flaky_data_fetch,
                endpoint,
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,          # 1s, 2s, 4s between retries
                    maximum_interval=timedelta(seconds=10),
                    maximum_attempts=5,               # Give up after 5 total attempts
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )
            workflow.logger.info(f"Data fetched after {raw_data.get('attempt')} attempt(s)")

            # Simulate a "resource" being allocated
            allocated_resource = f"resource-for-{endpoint}"

            # ── Step 2: Validate (non-retryable if invalid) ───────────────────
            workflow.logger.info("Step 2: Validating data...")
            try:
                validated = await workflow.execute_activity(
                    validate_data,
                    raw_data,
                    retry_policy=RetryPolicy(maximum_attempts=1),  # No retries for validation
                    start_to_close_timeout=timedelta(seconds=10),
                )
            except ActivityError as e:
                # Validation failed — non-retryable — do cleanup and stop
                workflow.logger.error(f"Validation failed: {e.cause}")
                await workflow.execute_activity(
                    compensate_cleanup,
                    allocated_resource,
                    start_to_close_timeout=timedelta(seconds=10),
                )
                return f"Workflow stopped: data validation failed. Cleanup done."

            # ── Step 3: Process with heartbeat ────────────────────────────────
            workflow.logger.info("Step 3: Processing items with heartbeat...")
            items = [f"item_{i}" for i in range(10)]
            processed = await workflow.execute_activity(
                process_with_heartbeat,
                items,
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=2),
                    maximum_attempts=3,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                heartbeat_timeout=timedelta(seconds=10),  # Must heartbeat every 10s
            )

            return (
                f"Workflow completed successfully!\n"
                f"  Endpoint: {endpoint}\n"
                f"  Fetched value: {validated['value']}\n"
                f"  Processed {len(processed)} items\n"
                f"  Raw fetch required {raw_data.get('attempt')} attempt(s) (auto-retried!)"
            )

        except ActivityError as e:
            # Unexpected activity failure — run compensation
            workflow.logger.error(f"Unexpected failure: {e}")
            if allocated_resource:
                await workflow.execute_activity(
                    compensate_cleanup,
                    allocated_resource,
                    retry_policy=RetryPolicy(maximum_attempts=5),
                    start_to_close_timeout=timedelta(seconds=30),
                )
            raise
