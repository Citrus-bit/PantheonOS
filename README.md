<div align="center">
  <h1> Pantheon </h1>

  <p> A framework for building distributed LLM-based multi-agent systems. </p>

  <p>
    <a href="https://github.com/aristoteleo/pantheon-agents/actions/workflows/test.yml">
        <img src="https://github.com/aristoteleo/pantheon-agents/actions/workflows/test.yml/badge.svg" alt="Build Status">
    </a>
    <a href="https://pypi.org/project/pantheon-agents/">
      <img src="https://img.shields.io/pypi/v/pantheon-agents.svg" alt="Install with PyPi" />
    </a>
    <a href="https://github.com/aristoteleo/pantheon-agents/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/aristoteleo/pantheon-agents" alt="MIT license" />
    </a>
  </p>
</div>

## Features

- **Multi-Agent Teams**: PantheonTeam, Sequential, Swarm, MoA, and AgentAsToolTeam patterns
- **Rich Toolsets**: File operations, Python/Shell execution, Jupyter notebooks, Web browsing, RAG
- **Interactive REPL**: Full-screen file viewer, syntax highlighting, approval workflows
- **MCP Integration**: Native Model Context Protocol support
- **Learning System**: Skillbook-based agent improvement
- **Distributed Architecture**: NATS-based communication for scalable deployments

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that handles dependencies efficiently.

```bash
# Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and install
git clone https://github.com/aristoteleo/pantheon-agents.git
cd pantheon-agents
uv sync

# With optional dependencies
uv sync --extra knowledge  # RAG/vector search support
uv sync --extra slack      # Slack integration
uv sync --extra r          # R language support (requires R installed)
```

### Using pip

```bash
# Basic installation
pip install pantheon-agents

# With optional dependencies
pip install "pantheon-agents[knowledge]"  # RAG/vector search support
pip install "pantheon-agents[slack]"      # Slack integration
```

### Development Installation

```bash
git clone https://github.com/aristoteleo/pantheon-agents.git
cd pantheon-agents
uv sync --extra dev --extra knowledge

# Run tests
uv run pytest tests/
```

## Requirements

- Python 3.10+
- API keys for LLM providers (e.g., `OPENAI_API_KEY` for OpenAI models)

## Quick Start

### Using the REPL

The easiest way to start:

```bash
# With uv
uv run python -m pantheon.repl

# Or with pip installation
python -m pantheon.repl
```

### Creating an Agent

```python
import asyncio
from pantheon import Agent

async def main():
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4o-mini"
    )
    await agent.chat()

asyncio.run(main())
```

### Using Toolsets

```python
from pantheon import Agent
from pantheon.toolsets import FileManagerToolSet, ShellToolSet

agent = Agent(
    name="developer",
    instructions="You are a developer assistant.",
    model="gpt-4o",
    tools=[FileManagerToolSet(), ShellToolSet()]
)
```

### Creating Teams

```python
from pantheon import Agent
from pantheon.team import PantheonTeam

researcher = Agent(name="researcher", instructions="Research topics.")
writer = Agent(name="writer", instructions="Write content.")

team = PantheonTeam([researcher, writer])
await team.chat()
```

## Documentation

See the [docs](docs/) folder for detailed documentation.

## Examples

See the [examples](examples/) folder for usage patterns and implementations.

## License

MIT License
