
# BankMate — Banking Customer Intelligence Assistant

BankMate is a GenAI-powered banking assistant built with Python, LangChain and Groq. It combines structured banking data retrieval with LLM-powered analysis and is being extended with Retrieval-Augmented Generation (RAG) for policy document search.

## Features

- LLM integration using Groq and GPT-OSS.
- Customer profile retrieval from CSV datasets.
- Transaction retrieval and analysis.
- Financial metric calculations using Python.
- LangChain tool calling and multi-tool orchestration.
- Modular application architecture.
- RAG-based banking policy retrieval (in development).

## Tech Stack

- Python
- LangChain
- Groq API
- GPT-OSS 20B
- ChromaDB
- Sentence Transformers
- PyTorch

## Project Structure

```text
bankmate/
├── app/
├── data/
├── knowledge/
│   └── policies/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd bankmate
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Add your Groq API key to `.env`.

### 5. Run the application

```bash
python main.py
```

## Current Capabilities

BankMate can retrieve customer information, fetch transaction records, calculate transaction metrics and answer questions using LangChain tool calling.

## Roadmap

- [x] Initial LLM integration
- [x] Customer and transaction data integration
- [x] LangChain tool calling
- [x] Modular application structure
- [ ] Document ingestion and chunking
- [ ] Embedding generation
- [ ] ChromaDB vector storage
- [ ] RAG-based policy retrieval
- [ ] Interactive application interface
- [ ] Evaluation and production-readiness improvements

## Dataset Disclaimer

This project uses synthetic banking data and fictional policy documents for educational and development purposes. It does not contain real customer records or represent actual banking policies.

## Security

Never commit API keys, credentials or confidential customer information.
