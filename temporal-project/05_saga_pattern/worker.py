import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from activities import (
    reserve_inventory, charge_payment, create_shipment,
    release_inventory, refund_payment, cancel_shipment,
)
from workflow import OrderSagaWorkflow

TASK_QUEUE = "saga-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[OrderSagaWorkflow],
        activities=[
            reserve_inventory, charge_payment, create_shipment,
            release_inventory, refund_payment, cancel_shipment,
        ],
    )
    print(f"Saga Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
