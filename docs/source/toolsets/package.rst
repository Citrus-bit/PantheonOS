PackageToolSet
==============

The PackageToolSet provides package and tool discovery capabilities with keyword and semantic search support for Pantheon workspaces.

Overview
--------

Key features:

* **Package Discovery**: List and search available packages
* **Tool Search**: Find tools using keyword or semantic search
* **LLM-Based Matching**: Intelligent semantic search using LLM
* **MCP Integration**: Works with MCP-based packages

Basic Usage
-----------

.. code-block:: python

   from pantheon.agent import Agent
   from pantheon.toolsets import PackageToolSet

   # Create package toolset
   package_tools = PackageToolSet(
       name="packages",
       workdir="/path/to/workspace",
       enable_semantic_search=True
   )

   # Create agent and add toolset at runtime
   agent = Agent(
       name="assistant",
       instructions="You can find and use available tools and packages.",
       model="gpt-4o"
   )
   await agent.toolset(package_tools)

   await agent.chat()

Constructor Parameters
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Parameter
     - Type
     - Description
   * - ``name``
     - str
     - Name of the toolset
   * - ``workdir``
     - str | Path | None
     - Workspace root directory. Packages are stored in ``.pantheon/packages``.
   * - ``enable_semantic_search``
     - bool
     - Enable LLM-based semantic search (default: True)
   * - ``semantic_model``
     - str
     - Model for semantic search (default: "low" for cost efficiency)

Tools Reference
---------------

search_tools
~~~~~~~~~~~~

Search for extended tools using keyword or semantic search.

.. code-block:: python

   # Semantic search (default)
   result = await package_tools.search_tools(
       query="convert CSV to Excel",
       semantic=True,
       top_k=5
   )

   # Keyword search
   result = await package_tools.search_tools(
       query="export",
       semantic=False,
       top_k=10
   )

   # List all tools
   result = await package_tools.search_tools()

**Parameters:**

- ``query``: Search query (natural language or keywords). None returns all tools.
- ``semantic``: Use LLM-based semantic search (default: True)
- ``top_k``: Maximum number of results
- ``use_context``: Use conversation history for context-aware search

**Returns:**

.. code-block:: python

   {
       "success": True,
       "tools": [
           {
               "package": "data_tools",
               "method": "csv_to_excel",
               "signature": "(input_path: str, output_path: str)",
               "doc": "Convert CSV file to Excel format",
               "async": True,
               "call_example": "await packages.data_tools.csv_to_excel(...)"
           }
       ]
   }

search_packages (Internal)
~~~~~~~~~~~~~~~~~~~~~~~~~~

List and search packages (not exposed to LLM, used by UI/API):

.. code-block:: python

   # List all packages
   result = await package_tools.search_packages()

   # Filter by keyword
   result = await package_tools.search_packages(query="data")

**Returns:**

.. code-block:: python

   {
       "success": True,
       "packages": [
           {
               "name": "data_tools",
               "description": "Data processing utilities",
               "methods": ["csv_to_excel", "json_to_yaml"],
               "status": "ready",
               "origin": "user"
           }
       ]
   }

Using Discovered Tools
----------------------

After finding tools, use them in Python code:

.. code-block:: python

   from pantheon.packages import packages as pp

   # Interactive (Notebook/IPython)
   result = await pp.packages.data_tools.csv_to_excel(
       input_path="data.csv",
       output_path="data.xlsx"
   )

   # Standalone script
   import asyncio

   async def main():
       return await pp.packages.data_tools.csv_to_excel(
           input_path="data.csv",
           output_path="data.xlsx"
       )

   result = asyncio.run(main())

Package API
-----------

Discover packages directly using the package API:

.. code-block:: python

   from pantheon.packages import packages as pp

   # List all packages
   packages = await pp.packages.list_packages()

   # Get package details
   info = pp.packages.describe("data_tools")

Examples
--------

Finding Data Processing Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find tools for data conversion
   result = await package_tools.search_tools(
       query="convert data between formats like CSV, JSON, Excel"
   )

   for tool in result["tools"]:
       print(f"{tool['package']}.{tool['method']}: {tool['doc']}")

Context-Aware Search
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After discussing a task in conversation
   result = await package_tools.search_tools(
       query="find something similar to what we discussed",
       use_context=True  # Uses conversation history
   )

Agent-Driven Tool Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pantheon.agent import Agent
   from pantheon.toolsets import PackageToolSet

   package_tools = PackageToolSet(name="packages")

   assistant = Agent(
       name="assistant",
       instructions="""When users need functionality:
       1. Use search_tools to find relevant packages
       2. Explain what tools are available
       3. Show how to use them with code examples""",
       model="gpt-4o"
   )
   await assistant.toolset(package_tools)

   result = await assistant.run(
       "I need to process some CSV files and convert them to different formats"
   )

Best Practices
--------------

1. **Use semantic search**: Better results for natural language queries
2. **Combine with code**: Build complete programs using discovered tools
3. **Check async status**: Use ``await`` for async tools
4. **Limit results**: Use ``top_k`` for focused results
5. **Explore packages**: Use ``describe()`` for detailed method info
