from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities import say_hello, say_goodbye


@workflow.defn
class HelloWorldWorkflow:
    """
    Project 01: Hello World — The simplest Temporal workflow.

    Demonstrates:
    - Workflow definition with @workflow.defn
    - Activity execution with execute_activity
    - Sequential activity chaining
    - start_to_close_timeout (activity timeout)
    """

    @workflow.run
    async def run(self, name: str) -> str:
        workflow.logger.info(f"HelloWorldWorkflow started for: {name}")

        greeting = await workflow.execute_activity(
            say_hello,
            name,
            start_to_close_timeout=timedelta(seconds=10),
        )

        farewell = await workflow.execute_activity(
            say_goodbye,
            name,
            start_to_close_timeout=timedelta(seconds=10),
        )

        return f"{greeting}\n{farewell}"
