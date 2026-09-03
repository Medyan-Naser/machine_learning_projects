from temporalio import activity


@activity.defn
async def say_hello(name: str) -> str:
    return f"Hello, {name}! Welcome to Temporal!"


@activity.defn
async def say_goodbye(name: str) -> str:
    return f"Goodbye, {name}! Thanks for using Temporal!"
