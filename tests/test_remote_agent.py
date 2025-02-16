import asyncio
from pantheon.agent import Agent, AgentResponse
from pantheon.remote.agent import AgentService, RemoteAgent


async def test_remote_agent():
    agent = Agent(
        "scifi_fan",
        "You are a scifi fan.",
        model="gpt-4o-mini",
    )
    service = AgentService(agent)
    service_task = asyncio.create_task(service.run())
    await asyncio.sleep(1.0)
    remote_agent = RemoteAgent(service.worker.service_id)
    res = await remote_agent.run("What is the best scifi book?")
    assert isinstance(res, AgentResponse)
    service_task.cancel()
