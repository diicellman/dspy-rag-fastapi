# FastAPI Wrapper for DSPy

## Introduction

This project is a [FastAPI](https://github.com/tiangolo/fastapi) wrapper designed to integrate with the [DSPy](https://github.com/stanfordnlp/dspy) framework developed by StanfordNLP, offering a straightforward example of building a FastAPI backend with DSPy capabilities. Uniquely, this implementation is fully local, utilizing [Ollama](https://github.com/ollama/ollama) for both the language and embedding models, and [Chroma DB](https://github.com/chroma-core/chroma) for vector storage. This setup ensures that all operations, from querying to data storage, are performed on the local machine without the need for external cloud services, enhancing privacy and data security.

## Features

- **Local Execution**: Everything runs on your local machine, ensuring data privacy and security. No external cloud services are involved.
- **Ollama Integration**: Leverages Ollama with the phi-2 language model and nomic embedding model.
- **Chroma DB for Vector Storage**: Uses Chroma DB for efficient and scalable vector storage, facilitating fast and accurate retrieval of information.
- **Zero-shot-query**: Allows users to perform zero-shot queries using DSPy through a simple GET request.
- **Compiled-query**: Enables the compilation of queries for optimized execution, accessible via GET.
- **Compile-program**: Offers an interface for compiling DSPy programs through a POST request, facilitating more complex interactions with the language model.

## Architecture

The FastAPI wrapper integrates DSPy with Ollama (utilizing the phi-2 language model and nomic embedding model) and Chroma DB in a seamless manner, providing a robust backend for applications requiring advanced natural language processing and data retrieval capabilities. Here's how the components interact within our local setup:

- **DSPy Framework**: Handles the optimization of language model prompts and weights, offering a sophisticated interface for programming with language models.
- **Ollama**: Serves as the backend for both the language model (phi-2) and the embedding model (nomic), enabling powerful and efficient natural language understanding and generation.
- **Chroma DB**: Acts as the vector store, allowing for efficient storage and retrieval of high-dimensional data vectors, which is crucial for tasks such as semantic search and similarity matching.

This local setup not only enhances data security and privacy but also provides developers with a flexible and powerful environment for building advanced NLP applications.

## Installation

### Prerequisites

- Docker
- Git (optional, for cloning the repository)
- Ollama,  follow the [readme](https://github.com/ollama/ollama) to set up and run a local Ollama instance.

### Clone the Repository

First, clone the repository to your local machine (skip this step if you have the project files already).

```bash
git clone https://github.com/diicellman/dspy-rag-fastapi.git
cd dspy-rag-fastapi
```
### Getting Started with Local Development

First, setup the environment:

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell
```
Then run this command:
```bash
python main.py
```
### Getting Started with Docker
This project is containerized with Docker, allowing for easy setup and deployment. To build and run the Docker container, use the following commands from the root of the project directory:

```bash
docker build -t dspy-fastapi-wrapper .
docker run -d --name dspy-wrapper -p 8000:8000 dspy-fastapi-wrapper
```
This will build the Docker image for the FastAPI wrapper and run it as a container, exposing the application on port 8000 of your host machine.

## Usage

After starting the FastAPI server, you can interact with the API endpoints as follows:

| Method | Endpoint             | Description                        | Example                                                                                      |
|--------|----------------------|------------------------------------|----------------------------------------------------------------------------------------------|
| GET    | `/zero-shot-query`   | Perform a zero-shot query.         | `curl http://localhost:8000/api/rag/zero-shot-query?query=<your-query>`                                   |
| GET    | `/compiled-query`    | Get a compiled query.              | `curl http://localhost:8000/api/rag/compiled-query?query=<your-query>`                                    |
| POST   | `/compile-program`   | Compile a DSPy program.            | `curl -X POST http://localhost:8000/api/rag/compile-program -H "Content-Type: application/json" -d '{"program": "<your-program>"}'` |

Ensure to replace `<your-query>` and `<your-program>` with the actual query and DSPy program you wish to execute.


## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs, questions, and suggestions.

