import asyncio

from app.outbound.followup_runner import run_followups


if __name__ == "__main__":
    asyncio.run(run_followups())
