Team
====

This section covers the various team collaboration patterns available in Pantheon. Teams enable multiple agents to work together effectively on complex tasks.

.. toctree::
   :maxdepth: 1
   
   sequential_team
   moa_team
   swarm_team
   swarm_center_team
   pantheon_team

Overview
--------

Pantheon provides several team structures to match different collaboration needs:

- **Sequential Team**: Agents work in a predefined order
- **MoA Team**: Multiple agents work independently, then synthesize
- **Swarm Team**: Dynamic agent selection based on needs
- **SwarmCenter Team**: Central coordinator manages worker agents
- **Pantheon Team**: Advanced hybrid team structure

Choosing a Team Type
--------------------

Sequential Team
~~~~~~~~~~~~~~~

Best for:
- Multi-step workflows
- Pipeline processing
- Tasks with clear dependencies

.. code-block:: python

   from pantheon.team import SequentialTeam
   
   team = SequentialTeam([researcher, analyst, writer])

MoA (Mixture of Agents) Team
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Best for:
- Getting diverse perspectives
- Robust decision making
- Complex problem solving

.. code-block:: python

   from pantheon.team import MoATeam
   
   team = MoATeam(
       agents=[expert1, expert2, expert3],
       aggregator=synthesizer
   )

Swarm Team
~~~~~~~~~~

Best for:
- Dynamic workflows
- Flexible task routing
- Adaptive problem solving

.. code-block:: python

   from pantheon.team import SwarmTeam
   
   team = SwarmTeam([generalist, specialist1, specialist2])

SwarmCenter Team
~~~~~~~~~~~~~~~~

Best for:
- Complex coordination
- Task distribution
- Hierarchical workflows

.. code-block:: python

   from pantheon.team import SwarmCenterTeam
   
   team = SwarmCenterTeam(
       center=coordinator,
       workers=[worker1, worker2, worker3]
   )

Quick Example
-------------

Here's a simple example of creating and using a team:

.. code-block:: python

   from pantheon.agent import Agent
   from pantheon.team import SequentialTeam
   
   # Create specialized agents
   researcher = Agent(
       name="researcher",
       instructions="Research the topic thoroughly."
   )
   
   writer = Agent(
       name="writer",
       instructions="Write clear, engaging content."
   )
   
   editor = Agent(
       name="editor",
       instructions="Polish and improve the text."
   )
   
   # Create team
   content_team = SequentialTeam([researcher, writer, editor])
   
   # Use team
   result = await content_team.run("Write about AI safety")

Team Features
-------------

All teams support:

- **Message passing**: Automatic context sharing between agents
- **Memory sharing**: Optional shared memory across team members
- **Error handling**: Graceful handling of agent failures
- **Streaming**: Real-time response streaming
- **Monitoring**: Track team performance and agent interactions

Best Practices
--------------

1. **Clear Roles**: Define distinct responsibilities for each agent
2. **Appropriate Size**: Keep teams between 3-7 agents for efficiency
3. **Error Recovery**: Implement fallback strategies
4. **Testing**: Test team dynamics with various scenarios
5. **Monitoring**: Track performance and optimize