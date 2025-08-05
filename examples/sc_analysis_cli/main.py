import fire
from pantheon.toolsets.scraper import ScraperToolSet
from pantheon.toolsets.shell import ShellToolSet
from pantheon.toolsets.vector_rag import VectorRAGToolSet
from pantheon.agent import Agent


async def main(path_to_rag_db: str):
    scraper_toolset = ScraperToolSet("scraper")
    shell_toolset = ShellToolSet("shell")
    vector_rag_toolset = VectorRAGToolSet(
        "vector_rag",
        db_path=path_to_rag_db,
    )

    instructions = """
    You are a CLI assistant that can run perfrom the Single-Cell/Spatial genomics upstream analysis.
    You can run shell commands to perform the analysis.
    Given the user's input, you should first analyze the input and determine your analysis plan.
    Then, you should output the analysis plan with check boxes for each step.
    You can search the vector database to get the knowledge about the tools.
    If you didn't find the information you need, you can search the web,
    you can use google search or web crawl from the scraper toolset.
    Then, you should run the analysis step by step with the shell tools.
    After all the analysis is done, you should output the analysis results and summarize the results.
    """

    agent = Agent(
        "sc_cli_bot",
        instructions,
        model="gpt-4.1",
    )
    agent.toolset(scraper_toolset)
    agent.toolset(shell_toolset)
    agent.toolset(vector_rag_toolset)

    await agent.chat()


if __name__ == "__main__":
    fire.Fire(main)