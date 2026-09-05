import asyncio
from temporalio.client import Client

from workflow import OrderSagaWorkflow

TASK_QUEUE = "saga-queue"


async def run_order(client, order_id: str, items: list[dict], total: float, label: str):
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  Order ID: {order_id}, Total: ${total:.2f}")
    result = await client.execute_workflow(
        OrderSagaWorkflow.run,
        args=[order_id, "customer-001", items, total],
        id=order_id,
        task_queue=TASK_QUEUE,
    )
    print(f"\nResult:\n{result}")


async def main():
    client = await Client.connect("localhost:7233")

    items_normal = [
        {"product_id": "LAPTOP_X1", "quantity": 1, "price": 999.99},
        {"product_id": "MOUSE_USB", "quantity": 2, "price": 29.99},
    ]

    items_oos = [
        {"product_id": "FAIL_PRODUCT", "quantity": 1, "price": 500.00},
    ]

    print("Saga Pattern Demo — Order Processing")
    print("Check Temporal UI at: http://localhost:8233\n")

    # Scenario 1: Happy path — everything succeeds
    await run_order(
        client,
        order_id="order-saga-success-001",
        items=items_normal,
        total=1059.97,
        label="SCENARIO 1: Happy Path (all steps succeed)",
    )

    # Scenario 2: Out-of-stock — inventory fails, no compensations needed (nothing happened yet)
    await run_order(
        client,
        order_id="order-saga-oos-001",
        items=items_oos,
        total=500.00,
        label="SCENARIO 2: Out of Stock (inventory fails → no compensations needed)",
    )

    # Scenario 3: Payment limit exceeded — inventory reserved then refunded
    await run_order(
        client,
        order_id="order-saga-payment-001",
        items=items_normal,
        total=99999.99,  # Exceeds payment limit
        label="SCENARIO 3: Payment Failure (inventory reserved → then released as compensation)",
    )

    print(f"\n{'='*60}")
    print("All scenarios complete!")
    print("Check the Temporal UI at: http://localhost:8233")
    print("  → Compare the event histories of all 3 workflows")
    print("  → See compensation activities in the failed scenarios")


if __name__ == "__main__":
    asyncio.run(main())
