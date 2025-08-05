RAG System (Auto-Build)
=======================

The RAG System provides tools for automatically building vector databases from various sources including documentation websites, GitHub repositories, and PDF files. It features web crawling, document processing, and integration with Hugging Face for database distribution.

Overview
--------

The RAG auto-build system (``pantheon.toolsets.utils.rag``) provides:

- **Automated Web Crawling**: Deep crawl documentation sites and extract content
- **Multi-Source Support**: Build from websites, GitHub READMEs, PDFs, and local files
- **Vector Database Creation**: Automatic chunking and embedding with LanceDB
- **Hugging Face Integration**: Upload and download pre-built databases
- **Caching System**: Intelligent caching for embeddings and build progress
- **YAML Configuration**: Define sources and parameters in YAML files

YAML Configuration Format
-------------------------

The YAML configuration file defines one or more vector databases with their sources:

Basic Structure
~~~~~~~~~~~~~~~

::

    database_name:
      description: Description of the database
      type: vector_db
      parameters:
        embedding_model: text-embedding-3-large
        chunk_size: 4000
        chunk_overlap: 200
      items:
        item_name:
          type: source_type
          url: source_url
          description: Item description

Supported Source Types
~~~~~~~~~~~~~~~~~~~~~~

- **package documentation**: Deep crawls documentation websites
- **tutorial**: Processes tutorial websites with multi-page content
- **github readme**: Fetches README files from GitHub repositories
- **pdf**: Downloads and processes PDF documents

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

::

    single-cell-python-packages:
      description: Vector database of single-cell python packages documentation
      type: vector_db
      parameters:
        embedding_model: text-embedding-3-large
        chunk_size: 4000
        chunk_overlap: 200
      items:
        scanpy:
          type: package documentation
          url: https://scanpy.readthedocs.io/en/stable/
          description: Scanpy toolkit for single-cell analysis
        sc-best-practices:
          type: tutorial
          url: https://www.sc-best-practices.org/
          description: Best practices guide for single-cell analysis
        star:
          type: pdf
          url: https://raw.githubusercontent.com/alexdobin/STAR/master/doc/STARmanual.pdf
          description: STAR RNA-seq aligner manual

Command Line Usage
------------------

Build from YAML Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build the database::

    python -m pantheon.toolsets.utils.rag build config.yaml ./output_dir

Upload to Hugging Face
~~~~~~~~~~~~~~~~~~~~~~

Share your built database::

    # Set your Hugging Face token
    export HUGGINGFACE_TOKEN=your_token_here
    
    # Upload to default repo (NaNg/pantheon_rag_db)
    python -m pantheon.toolsets.utils.rag upload ./output_dir
    
    # Or specify custom repo
    python -m pantheon.toolsets.utils.rag upload ./output_dir --repo-id your-username/your-repo

Download Pre-built Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download existing databases::

    # Download from default repo
    python -m pantheon.toolsets.utils.rag download ./local_dir
    
    # Download from custom repo
    python -m pantheon.toolsets.utils.rag download ./local_dir --repo-id username/repo --filename custom.zip

Build Process Details
---------------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

After building, the output directory contains::

    output_dir/
    └── database_name/
        ├── metadata.yaml        # Database configuration
        ├── info_cache.json      # Build progress tracking
        ├── raw/                 # Downloaded raw content
        │   └── item_name/
        │       ├── *.md        # Markdown files
        │       └── *.pdf       # PDF files
        └── lancedb/            # Vector database files
            └── database_name.lance/

Processing Pipeline
~~~~~~~~~~~~~~~~~~~

1. **Download Phase**:
   
   - For documentation/tutorials: Deep crawl with configurable depth
   - For GitHub READMEs: Direct file download
   - For PDFs: Binary file download

2. **Content Extraction**:
   
   - HTML → Markdown conversion for web content
   - PDF text extraction using PyMuPDF
   - Duplicate removal via content hashing

3. **Text Processing**:
   
   - Smart chunking with configurable size and overlap
   - Metadata preservation (source, URL)
   - Context maintenance across chunks

4. **Vector Storage**:
   
   - Embedding generation via OpenAI API
   - LanceDB storage with PyArrow schema
   - Metadata indexing for filtering

Progress Tracking
~~~~~~~~~~~~~~~~~

The system maintains build state in ``info_cache.json``::

    {
        "item_name": {
            "success": true,
            "created_at": "2024-01-01T12:00:00",
            "download_success": true
        }
    }

Key behaviors:

- Successfully cached items (``success: true``) are skipped on subsequent builds
- Failed items can be retried without re-processing successful ones
- To force re-download: Set ``download_success`` to ``false`` in ``info_cache.json``
- To force complete rebuild: Set ``success`` to ``false`` in ``info_cache.json``
- Then re-run the build command to process the modified items

Programmatic Usage
------------------

Using VectorDB Class
~~~~~~~~~~~~~~~~~~~~

Direct database operations::

    from pantheon.toolsets.utils.rag.vectordb import VectorDB
    
    # Load existing database
    db = VectorDB("./output_dir/database_name")
    
    # Query the database
    results = await db.query(
        query="How to perform clustering analysis?",
        top_k=5,
        source="scanpy"  # Optional: filter by source
    )
    
    # Insert new content
    await db.insert(
        text="Your new content here",
        metadata={"source": "custom", "date": "2024-01-01"}
    )
    
    # Insert from file with automatic chunking
    await db.insert_from_file(
        file_path="./new_doc.md",
        metadata={"source": "local_docs"}
    )

Building Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~

Build databases from Python code::

    import asyncio
    from pantheon.toolsets.utils.rag.build import build_vector_db
    
    db_config = {
        "type": "vector_db",
        "parameters": {
            "embedding_model": "text-embedding-3-large",
            "chunk_size": 4000,
            "chunk_overlap": 200
        },
        "items": {
            "my_docs": {
                "type": "package documentation",
                "url": "https://mydocs.example.com",
                "description": "My documentation"
            },
            "manual": {
                "type": "pdf",
                "url": "https://example.com/manual.pdf",
                "description": "User manual"
            }
        }
    }
    
    asyncio.run(build_vector_db("my_knowledge_base", db_config, "./output"))

Full Build Example
~~~~~~~~~~~~~~~~~~

Complete workflow from YAML to usage::

    import asyncio
    from pantheon.toolsets.utils.rag.build import build_all
    
    # Build all databases defined in YAML
    asyncio.run(build_all("./config.yaml", "./output"))

Special Features
----------------

PDF Support
~~~~~~~~~~~

The system can process PDF documents::

    items:
      research_paper:
        type: pdf
        url: https://arxiv.org/pdf/2301.00001.pdf
        description: Research paper on transformers

PDFs are:

- Downloaded as binary files
- Text extracted using PyMuPDF
- Processed like other text documents
- Stored with original URL metadata

Web Crawling Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

For documentation and tutorial types:

- **max_depth**: Controls crawling depth (default: 1)
- **include_external**: Include external links (default: false)
- Uses BFS (Breadth-First Search) strategy
- Automatic markdown extraction from HTML
- Removes duplicate content via SHA-256 hashing

Embedding Models
~~~~~~~~~~~~~~~~

Supported OpenAI embedding models:

- ``text-embedding-3-large``: Best quality, 3072 dimensions
- ``text-embedding-3-small``: Faster, 1536 dimensions
- ``text-embedding-ada-002``: Legacy model, 1536 dimensions

Caching System
~~~~~~~~~~~~~~

Two-level caching for efficiency:

1. **Embedding Cache**: Disk-based cache prevents redundant API calls
2. **Build Cache**: ``info_cache.json`` tracks processing status

Hugging Face Integration
------------------------

Upload and Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

The system integrates with Hugging Face for sharing databases:

1. **Packaging**: Creates ZIP archive of entire database
2. **Upload**: Pushes to Hugging Face dataset repository
3. **Versioning**: Uploaded as ``latest.zip`` by default

::

    # Upload with authentication
    export HUGGINGFACE_TOKEN=hf_xxxxx
    python -m pantheon.toolsets.utils.rag upload ./my_db --repo-id myorg/my-rag-db

Download and Use
~~~~~~~~~~~~~~~~

Download pre-built databases::

    # Download and extract
    python -m pantheon.toolsets.utils.rag download ./local_db --repo-id myorg/my-rag-db
    
    # Use immediately
    from pantheon.toolsets.utils.rag.vectordb import VectorDB
    db = VectorDB("./local_db/database_name")

Real-World Examples
-------------------

Single-Cell Analysis Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a comprehensive single-cell analysis database::

    # single_cell.yaml
    single-cell-tools:
      description: Comprehensive single-cell analysis tools documentation
      type: vector_db
      parameters:
        embedding_model: text-embedding-3-large
        chunk_size: 4000
        chunk_overlap: 200
      items:
        scanpy:
          type: package documentation
          url: https://scanpy.readthedocs.io/en/stable/
          description: Core single-cell analysis toolkit
        scvi-tools:
          type: package documentation
          url: https://docs.scvi-tools.org/en/stable/
          description: Deep learning for single-cell
        cellranger:
          type: pdf
          url: https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/using/tutorial_ct
          description: 10x Genomics Cell Ranger manual

Machine Learning Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ML framework knowledge base::

    # ml_docs.yaml
    ml-frameworks:
      description: Machine learning frameworks documentation
      type: vector_db
      parameters:
        embedding_model: text-embedding-3-large
        chunk_size: 3000
        chunk_overlap: 300
      items:
        pytorch:
          type: package documentation
          url: https://pytorch.org/docs/stable/
          description: PyTorch deep learning framework
        tensorflow:
          type: package documentation
          url: https://www.tensorflow.org/api_docs
          description: TensorFlow ML platform
        papers:
          type: pdf
          url: https://arxiv.org/pdf/1706.03762.pdf
          description: Attention Is All You Need paper

Best Practices
--------------

1. **Choose Appropriate Chunk Sizes**: 
   - Larger chunks (4000-5000) for narrative content
   - Smaller chunks (1000-2000) for API references
   - Balance between context and precision

2. **Optimize Embedding Models**:
   - Use ``text-embedding-3-large`` for best quality
   - Consider ``text-embedding-3-small`` for large datasets
   - Monitor API costs

3. **Organize Sources Logically**: 
   - Group related documentation in same database
   - Use descriptive item names
   - Add clear descriptions for each source

4. **Handle Build Failures**:
   - Check ``info_cache.json`` for error details
   - Failed items can be retried individually
   - Network issues don't affect completed items

5. **Use Hugging Face for Distribution**: 
   - Share pre-built databases to save compute
   - Version control via repository tags
   - Collaborate on knowledge bases

6. **Regular Updates**: 
   - Rebuild periodically for fresh content
   - Use caching to minimize redundant work
   - Track changes via git

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Build Failures**:

- Check network connectivity for web crawling
- Verify URLs are accessible (not behind authentication)
- Review error messages in ``info_cache.json``
- Ensure sufficient disk space

**PDF Processing Errors**:

- Verify PDF URL is directly accessible
- Check if PDF requires authentication
- Some PDFs may have text extraction restrictions
- Install PyMuPDF: ``pip install pymupdf``

**Embedding Errors**:

- Ensure ``OPENAI_API_KEY`` is set correctly
- Check API rate limits and quotas
- Verify model name is correct
- Monitor token usage

**Storage Issues**:

- Ensure sufficient disk space (databases can be large)
- Check write permissions on output directory
- Clean old cache files if needed
- Verify LanceDB installation

Performance Tips
~~~~~~~~~~~~~~~~

- Use embedding cache to avoid redundant API calls
- Enable progress tracking to resume interrupted builds
- Process sources in parallel when system allows
- Consider smaller embedding models for very large datasets
- Use local caching for frequently accessed databases

Integration with VectorRAGToolSet
----------------------------------

The VectorRAGToolSet class provides a toolset interface for agents to interact with the built databases.

Basic Usage
~~~~~~~~~~~

Create a VectorRAGToolSet with your built database::

    from pantheon.toolsets.vector_rag import VectorRAGToolSet
    
    # Initialize with database path
    rag_toolset = VectorRAGToolSet(
        name="knowledge_assistant",
        db_path="./output/single-cell-tools"
    )
    
    # The toolset provides these tools:
    # - query_vector_db: Query the database with optional source filtering
    # - get_vector_db_info: Get metadata about the database

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Full configuration options::

    rag_toolset = VectorRAGToolSet(
        name="custom_rag",
        db_path="./output/ml-frameworks",
        worker_params=None,  # Optional worker configuration
        allow_insert=False,  # Enable insert_vector_db tool
        allow_delete=False,  # Enable delete_vector_db tool
        db_params={}        # Additional database parameters
    )

Available Tools
~~~~~~~~~~~~~~~

**query_vector_db**: Main query interface

- ``query``: Query string
- ``top_k``: Number of results (default: 3)
- ``source``: Optional source filter

**get_vector_db_info**: Returns database metadata including description

**insert_vector_db** (optional): Add new content
- Enabled with ``allow_insert=True``
- Parameters: ``text``, ``metadata``

**delete_vector_db** (optional): Remove content
- Enabled with ``allow_delete=True``
- Parameter: ``id`` (string or list)

Using with Agents
~~~~~~~~~~~~~~~~~

Connect the toolset to an agent as a remote service::

    from pantheon.agent import Agent
    from pantheon.toolsets.vector_rag import VectorRAGToolSet
    import asyncio
    
    async def main():
        # Create RAG toolset
        rag_toolset = VectorRAGToolSet(
            name="bio_knowledge",
            db_path="./output/single-cell-tools",
            allow_insert=True  # Allow agent to add new knowledge
        )
        
        # Start toolset service
        await rag_toolset.start_service()
        
        # Create agent
        agent = Agent(
            name="bioinformatics_expert",
            instructions="You are a single-cell analysis expert. Use get_vector_db_info first to understand the database, then query_vector_db to find relevant information.",
            model="gpt-4"
        )
        
        # Connect agent to toolset
        await agent.remote_toolset(rag_toolset.service_id)
        
        # Query through agent
        response = await agent.run("How do I perform trajectory inference?")
        
    asyncio.run(main())

Programmatic Direct Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the toolset directly without agents::

    async def search_knowledge():
        rag = VectorRAGToolSet(
            name="ml_rag",
            db_path="./ml_db/ml-frameworks"
        )
        
        # Get database info
        info = await rag.get_vector_db_info()
        print(f"Database: {info['description']}")
        
        # Query the database
        results = await rag.query_vector_db(
            query="transformer architecture",
            top_k=5,
            source="pytorch"  # Filter by source
        )
        
        for result in results:
            print(f"Score: {result['score']}")
            print(f"Text: {result['text'][:200]}...")
            print(f"Source: {result['metadata']['source']}")
            print("---")