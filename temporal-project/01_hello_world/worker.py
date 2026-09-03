import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from activities import say_hello, say_goodbye
from workflow import HelloWorldWorkflow

TASK_QUEUE = "hello-world-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[HelloWorldWorkflow],
        activities=[say_hello, say_goodbye],
    )
    print(f"Worker started — polling task queue: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
