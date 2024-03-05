# FastAPI Wrapper for DSPy

## Introduction

This project is a [FastAPI](https://github.com/tiangolo/fastapi) wrapper designed to integrate with the [DSPy](https://github.com/stanfordnlp/dspy) framework developed by StanfordNLP, offering a straightforward example of building a FastAPI backend with DSPy capabilities. Uniquely, this implementation is fully local, utilizing [Ollama](https://github.com/ollama/ollama) for both the language and embedding models, [Chroma DB](https://github.com/chroma-core/chroma) for vector storage, and [Arize Phoenix](https://github.com/Arize-ai/phoenix) for an observability layer. This setup ensures that all operations, from querying to data storage, are performed on the local machine without the need for external cloud services, enhancing privacy and data security.

## Features

- **Local Execution**: Everything runs on your local machine, ensuring data privacy and security. No external cloud services are involved.
- **Ollama Integration**: Leverages Ollama with the phi-2 language model and nomic embedding model by default. However, now with configurable LLM support, allowing users to specify the desired language model in the .env file or Docker Compose file.
- **Chroma DB for Vector Storage**: Uses Chroma DB for efficient and scalable vector storage, facilitating fast and accurate retrieval of information.
- **Arize Phoenix**: Incorporates Arize Phoenix for observability, offering real-time monitoring and analytics to track and improve model performance and system health.
- **Zero-shot-query**: Allows users to perform zero-shot queries using DSPy through a simple GET request.
- **Compiled-query**: Enables the compilation of queries for optimized execution, accessible via GET.
- **Compile-program**: Offers an interface for compiling DSPy programs through a POST request, facilitating more complex interactions with the language model.

## Architecture

The FastAPI wrapper integrates DSPy with Ollama, Arize Phoenix and Chroma DB in a seamless manner, providing a robust backend for applications requiring advanced natural language processing and data retrieval capabilities. Here's how the components interact within our local setup:

- **DSPy Framework**: Handles the optimization of language model prompts and weights, offering a sophisticated interface for programming with language models.
- **Ollama**: Serves as the backend for both the language model and the embedding model, enabling powerful and efficient natural language understanding and generation.
- **Chroma DB**: Acts as the vector store, allowing for efficient storage and retrieval of high-dimensional data vectors, which is crucial for tasks such as semantic search and similarity matching.
- **Arize Phoenix**: Phoenix makes your DSPy applications observable by visualizing the underlying structure of each call to your compiled DSPy module.

This local setup not only enhances data security and privacy but also provides developers with a flexible and powerful environment for building advanced NLP applications.

## Installation

### Prerequisites

- Docker and Docker-Compose
- Git (optional, for cloning the repository)
- Ollama,  follow the [readme](https://github.com/ollama/ollama) to set up and run a local Ollama instance.

### Clone the Repository

First, clone the repository to your local machine (skip this step if you have the project files already).

```bash
git clone https://github.com/diicellman/dspy-rag-fastapi.git
cd dspy-rag-fastapi
```
### Getting Started with Local Development

First, navigate to the backend directory:
```bash
cd backend/
```

Second, setup the environment:

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell
```
Specify your environment variables in an .env file in backend directory.
Example .env file:
```yml
ENVIRONMENT=<your_environment_value>
INSTRUMENT_DSPY=<true or false>
COLLECTOR_ENDPOINT=<your_arize_phoenix_endpoint>
OLLAMA_BASE_URL=<your_ollama_instance_endpoint>
OLLAMA_MODEL_NAME=<your_llm_model_name>
```
Third, run this command to create embeddings of data located in data/example folder:
```bash
python app/utils/load.py
```

Then run this command to start the FastAPI server:
```bash
python main.py
```
### Getting Started with Docker-Compose
This project now supports Docker Compose for easier setup and deployment, including backend services and Arize Phoenix for query tracing. 

1. Configure your environment variables in the .env file or modify the compose file directly.
2. Ensure that Docker is installed and running.
3. Run the command `docker-compose -f compose.yml up` to spin up services for the backend, and Phoenix.
4. Backend docs can be viewed using the [OpenAPI](http://0.0.0.0:8000/docs).
5. Traces can be viewed using the [Phoenix UI](http://0.0.0.0:6006).
7. When you're finished, run `docker compose down` to spin down the services.

## Usage

After starting the FastAPI server, you can interact with the API endpoints as follows:

| Method | Endpoint             | Description                        | Example                                                                                      |
|--------|----------------------|------------------------------------|----------------------------------------------------------------------------------------------|
| GET    | `/zero-shot-query`   | Perform a zero-shot query.         | `curl http://<your_address>:8000/api/rag/zero-shot-query?query=<your-query>`                                   |
| GET    | `/compiled-query`    | Get a compiled query.              | `curl http://<your_address>:8000/api/rag/compiled-query?query=<your-query>`                                    |
| POST   | `/compile-program`   | Compile a DSPy program.            | `curl -X POST http://<your_address>:8000/api/rag/compile-program -H "Content-Type: application/json" -d ''` |

Ensure to replace `<your-query>` and `<your-program>` with the actual query and DSPy program you wish to execute.


## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs, questions, and suggestions.

