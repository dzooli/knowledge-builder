# GitHub Copilot Context

This is the default Copilot prompt for this project.

## Tooling

- Git for source code versioning
- use Markdown for documentation and mkdocs for documentation generation

## Project Description

This project is an **automated ETL pipeline**: knowledge is extracted from documents OCR-ed by Paperless-ngx using the *
*Ollama** LLM, then loaded into a **Neo4j** graph via the official **Neo4j Memory MCP** server. The loading is performed
by a **LangChain Agent**; tool calls are **delegated to the LLM itself**. Optionally, the raw text can also be exported
to an **Obsidian** vault. It uses a scheduler to run the pipeline periodically and **verbose logging** using `loguru`.

## Guidelines

### Software design principles

As a professional software developer also experienced in test automation, you should follow
these rules:

- Use software design patterns
- Apply SOLID design principles as:
    - Single Responsibility Principle: A class should have only one reason to change, meaning it should have only one
      job.
    - Open/Closed Principle: Software entities should be open for extension but closed for modification.
    - Liskov Substitution Principle: Subtypes must be substitutable for their base types without altering the
      correctness of the program.
    - Interface Segregation Principle: Clients should not be forced to depend on interfaces they do not use.
    - Dependency Inversion Principle: High-level modules should not depend on low-level modules; both should depend on
      abstractions.
- Apply DRY principle
- Apply KISS principle
- Keep the method and function cognitive complexity below 15
- Keep method and function length below 30 lines

### Python instructions

- uv for package management using dependency groups
- pytest and pytest-cov for test automation and test coverage generation

#### Pythonic coding - applicable to Python

- Use slotted dataclasses or PydanticV2 data models when applicable
- Use generators and iterators when applicable
- Use effective Python data structures and algorithms
- Use enumerate instead of range
- Use type hints
- Use type annotations
- Use type checking
- Use type inference
- use del operator to clean up memory and force garbage collection when releasing large data structures
- use context managers
- use f-strings
- use list/set/dict comprehensions
- use built-in functions and libraries
- use exception handling
- use pathlib for file system paths
- use httpx for HTTP requests
- use 'yield from' structure when working with generators
