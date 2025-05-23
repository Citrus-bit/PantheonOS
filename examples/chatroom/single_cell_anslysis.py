import fire

from pantheon.chatroom.start import start_services
from pantheon.agent import Agent


async def agents_factory(endpoint):
    instructions = """You are the triage agent,
you should decide which agent to use based on the user's request.
If no related agent, you can do the task by yourself."""
    assistant_agent = Agent(
        name="Assistant",
        instructions=instructions,
        model="gpt-4.1",
        icon="🤖",
    )
    s = await endpoint.invoke("get_service", {"service_id_or_name": "python_interpreter"})
    if s is None:
        raise ValueError("Python interpreter service not found")
    await assistant_agent.remote_toolset(s["id"])

    s = await endpoint.invoke("get_service", {"service_id_or_name": "file_manager"})
    if s is None:
        raise ValueError("File manager service not found")
    await assistant_agent.remote_toolset(s["id"])

    s = await endpoint.invoke("get_service", {"service_id_or_name": "web_browse"})
    if s is None:
        raise ValueError("Web browser service not found")
    await assistant_agent.remote_toolset(s["id"])

    instructions = """You are a AI-agent for analyzing single-cell RNA-seq data.

Given a single-cell RNA-seq dataset,

you can write python code call scanpy package to analyze the data.

Basicly, given a single-cell RNA-seq dataset in h5ad / 10x format or other formats,
you should firstly output your plan and the code.
Then, you should execute the code to read the data,
then preprocess the data, and cluster the data, and finally visualize the data.
When you visualize the data, you should produce the publication level high-quality figures.

You can display the figures with it's path in markdown format.

After you ploted some figure, you should using view_image function to check the figure,
then according to the figure decide what you should do next.

After you finished the task, you should display the final result for user.

NOTE: Don't need to confirm with user at most time, just do the task.
"""

    single_cell_expert = Agent(
        name="Single cell expert",
        instructions=instructions,
        model="gpt-4.1",
        icon="🧬",
    )
    s = await endpoint.invoke("get_service", {"service_id_or_name": "single_cell_python_env"})
    if s is None:
        raise ValueError("Single cell python environment service not found")
    await single_cell_expert.remote_toolset(s["id"])

    s = await endpoint.invoke("get_service", {"service_id_or_name": "file_manager"})
    if s is None:
        raise ValueError("File manager service not found")
    await single_cell_expert.remote_toolset(s["id"])

    s = await endpoint.invoke("get_service", {"service_id_or_name": "web_browse"})
    if s is None:
        raise ValueError("Web browser service not found")
    await single_cell_expert.remote_toolset(s["id"])

    return {
        "triage": assistant_agent,
        "other": [single_cell_expert],
    }

async def main(endpoint_service_id: str, **kwargs):
    await start_services(
        service_name="pantheon-chatroom",
        memory_path="./.pantheon-chatroom",
        endpoint_service_id=endpoint_service_id,
        agents_factory=agents_factory,
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)