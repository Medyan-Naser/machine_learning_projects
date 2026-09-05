from datetime import timedelta
from typing import Callable
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from activities import (
        reserve_inventory,
        charge_payment,
        create_shipment,
        release_inventory,
        refund_payment,
        cancel_shipment,
    )


@workflow.defn
class OrderSagaWorkflow:
    """
    Project 05: Saga Pattern — Distributed Transaction with Compensations

    Demonstrates:
    - Saga pattern: each step has a compensating action
    - Compensation runs in reverse order on failure
    - ActivityError catching in the workflow
    - The "try-compensate" pattern

    Use Case:
    Order processing across inventory, payment, and shipping services.
    If payment fails after inventory is reserved, inventory must be released.
    If shipping fails after payment, payment must be refunded AND inventory released.

    Saga Steps:
      1. Reserve Inventory  → COMPENSATE: Release Inventory
      2. Charge Payment     → COMPENSATE: Refund Payment
      3. Create Shipment    → COMPENSATE: Cancel Shipment

    The compensations stack up as we progress. On failure, they run in reverse.
    """

    @workflow.run
    async def run(self, order_id: str, customer_id: str, items: list[dict], total: float) -> str:
        workflow.logger.info(f"Saga started for order: {order_id}, total: ${total:.2f}")

        # Stack of compensation functions (LIFO)
        compensations: list[tuple[Callable, list]] = []

        activity_opts = dict(
            retry_policy=RetryPolicy(maximum_attempts=3),
            start_to_close_timeout=timedelta(seconds=30),
        )

        try:
            # ── Step 1: Reserve Inventory ──────────────────────────────────────
            workflow.logger.info("Saga Step 1: Reserving inventory...")
            reservation_id = await workflow.execute_activity(
                reserve_inventory,
                args=[order_id, items],
                **activity_opts,
            )
            # Push compensation: if later steps fail, release this reservation
            compensations.append((release_inventory, [order_id, reservation_id]))
            workflow.logger.info(f"✓ Inventory reserved: {reservation_id}")

            # ── Step 2: Charge Payment ─────────────────────────────────────────
            workflow.logger.info("Saga Step 2: Charging payment...")
            transaction_id = await workflow.execute_activity(
                charge_payment,
                args=[order_id, customer_id, total],
                **activity_opts,
            )
            # Push compensation: if shipping fails, refund this payment
            compensations.append((refund_payment, [order_id, transaction_id]))
            workflow.logger.info(f"✓ Payment charged: {transaction_id}")

            # ── Step 3: Create Shipment ────────────────────────────────────────
            workflow.logger.info("Saga Step 3: Creating shipment...")
            shipment_id = await workflow.execute_activity(
                create_shipment,
                args=[order_id, customer_id],
                **activity_opts,
            )
            # Push compensation (in case future steps fail)
            compensations.append((cancel_shipment, [order_id, shipment_id]))
            workflow.logger.info(f"✓ Shipment created: {shipment_id}")

            # ── All steps succeeded ────────────────────────────────────────────
            return (
                f"Order {order_id} completed successfully!\n"
                f"  Customer: {customer_id}\n"
                f"  Total: ${total:.2f}\n"
                f"  Reservation: {reservation_id}\n"
                f"  Transaction: {transaction_id}\n"
                f"  Shipment: {shipment_id}"
            )

        except ActivityError as e:
            workflow.logger.error(f"Saga failure at step — running compensations: {e.cause}")

            # ── Run compensations in REVERSE order ─────────────────────────────
            for comp_activity, comp_args in reversed(compensations):
                try:
                    await workflow.execute_activity(
                        comp_activity,
                        args=comp_args,
                        retry_policy=RetryPolicy(maximum_attempts=5),  # Compensations must succeed
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                    workflow.logger.info(f"[COMPENSATION] {comp_activity.__name__} succeeded")
                except ActivityError as comp_err:
                    workflow.logger.error(
                        f"[COMPENSATION FAILED] {comp_activity.__name__}: {comp_err.cause}"
                    )

            return (
                f"Order {order_id} FAILED — all compensations applied.\n"
                f"  Failure reason: {e.cause}\n"
                f"  Compensations run: {[c[0].__name__ for c in reversed(compensations)]}"
            )
