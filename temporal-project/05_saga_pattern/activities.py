import time
from dataclasses import dataclass
from temporalio import activity
from temporalio.exceptions import ApplicationError


@dataclass
class OrderItem:
    product_id: str
    quantity: int
    price: float


@dataclass
class Order:
    order_id: str
    customer_id: str
    items: list[OrderItem]
    total: float


# ─── Forward (Business) Activities ─────────────────────────────────────────────

@activity.defn
async def reserve_inventory(order_id: str, items: list[dict]) -> str:
    """Reserve inventory for the order."""
    activity.logger.info(f"Reserving inventory for order {order_id}: {items}")
    time.sleep(0.3)

    # Simulate inventory check failure for demo purposes
    for item in items:
        if item.get("product_id") == "FAIL_PRODUCT":
            raise ApplicationError(f"Product {item['product_id']} is out of stock!", non_retryable=True)

    reservation_id = f"reservation-{order_id}"
    activity.logger.info(f"Inventory reserved: {reservation_id}")
    return reservation_id


@activity.defn
async def charge_payment(order_id: str, customer_id: str, amount: float) -> str:
    """Charge the customer's payment method."""
    activity.logger.info(f"Charging ${amount:.2f} for order {order_id}, customer {customer_id}")
    time.sleep(0.3)

    # Simulate payment failure for demo
    if amount > 10000:
        raise ApplicationError(f"Payment declined: amount ${amount} exceeds limit", non_retryable=True)

    transaction_id = f"txn-{order_id}-{int(amount * 100)}"
    activity.logger.info(f"Payment charged: {transaction_id}")
    return transaction_id


@activity.defn
async def create_shipment(order_id: str, customer_id: str) -> str:
    """Create a shipment record and schedule delivery."""
    activity.logger.info(f"Creating shipment for order {order_id}")
    time.sleep(0.3)

    shipment_id = f"ship-{order_id}"
    activity.logger.info(f"Shipment created: {shipment_id}")
    return shipment_id


# ─── Compensation (Rollback) Activities ────────────────────────────────────────

@activity.defn
async def release_inventory(order_id: str, reservation_id: str) -> None:
    """COMPENSATION: Release the reserved inventory."""
    activity.logger.info(f"[COMPENSATION] Releasing inventory {reservation_id} for order {order_id}")
    time.sleep(0.2)
    activity.logger.info(f"[COMPENSATION] Inventory released: {reservation_id}")


@activity.defn
async def refund_payment(order_id: str, transaction_id: str) -> None:
    """COMPENSATION: Refund the charged payment."""
    activity.logger.info(f"[COMPENSATION] Refunding transaction {transaction_id} for order {order_id}")
    time.sleep(0.2)
    activity.logger.info(f"[COMPENSATION] Refund issued: {transaction_id}")


@activity.defn
async def cancel_shipment(order_id: str, shipment_id: str) -> None:
    """COMPENSATION: Cancel the created shipment."""
    activity.logger.info(f"[COMPENSATION] Cancelling shipment {shipment_id} for order {order_id}")
    time.sleep(0.2)
    activity.logger.info(f"[COMPENSATION] Shipment cancelled: {shipment_id}")
