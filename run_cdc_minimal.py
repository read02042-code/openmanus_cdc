import argparse
import asyncio

from app.agent.cdc_plan import CDCPlanAgent
from app.logger import logger


async def main():
    parser = argparse.ArgumentParser(description="Run CDCPlan agent")
    parser.add_argument("--prompt", type=str, required=False, help="Input prompt")
    args = parser.parse_args()

    agent = CDCPlanAgent()
    try:
        prompt = args.prompt if args.prompt else input("Enter CDC event details: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return
        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
