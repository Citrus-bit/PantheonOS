import asyncio

from pantheon.agent import Agent
from pantheon.team import SwarmCenterTeam
from pantheon.repl.team import Repl


async def main():
    scifi_fan = Agent(
        name="Scifi Fan",
        instructions="You are a scifi fan.",
        model="gpt-4o-mini",
    )

    romance_fan = Agent(
        name="Romance Fan",
        instructions="You are a romance fan.",
        model="gpt-4o-mini",
    )

    triage = Agent(
        name="Triage",
        instructions="You are a triage agent.",
        model="gpt-4o-mini",
    )

    team = SwarmCenterTeam(triage, [scifi_fan, romance_fan])
    repl = Repl(team)
    await repl.run()


if __name__ == "__main__":
    asyncio.run(main())
