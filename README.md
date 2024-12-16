
# RAG Pipeline for Chatting with PDF and Websites

This repository demonstrates the implementation of a **Retrieval-Augmented Generation (RAG)** pipeline designed to interact with semi-structured data in **PDF files** and **websites**. The system can efficiently extract, chunk, embed, and store data for accurate query responses and comparisons. By leveraging **vector embeddings** and a **pre-trained LLM (Large Language Model)**, it generates detailed, context-rich answers to user queries.

## Table of Contents
- [Overview](#overview)
- [Task 1: Chat with PDF Using RAG Pipeline](#task-1-chat-with-pdf-using-rag-pipeline)
  - [Data Ingestion](#data-ingestion)
  - [Query Handling](#query-handling)
  - [Comparison Queries](#comparison-queries)
  - [Response Generation](#response-generation)
- [Task 2: Chat with Website Using RAG Pipeline](#task-2-chat-with-website-using-rag-pipeline)
  - [Data Ingestion](#data-ingestion-1)
  - [Query Handling](#query-handling-1)
  - [Response Generation](#response-generation-1)
- [Installation](#installation)
- [Usage](#usage)

## Overview

The goal of this repository is to implement an advanced RAG pipeline that allows users to interact with **PDF files** and **websites** to extract structured and unstructured data. The system uses embeddings to store the data and respond to user queries effectively by retrieving relevant data from a **vector database** and generating responses through a **pre-trained LLM**.

The pipeline is broken down into two core tasks:

1. **Chat with PDF**: Enables users to ask questions and retrieve information from semi-structured data in PDFs.
2. **Chat with Websites**: Allows users to query structured and unstructured data from websites by crawling and scraping the content.

## Task 1: Chat with PDF Using RAG Pipeline

This task involves processing PDF files containing semi-structured data, chunking the information, embedding it for efficient retrieval, and generating responses to user queries.

### Data Ingestion
- **Input**: PDF files containing structured or semi-structured data.
- **Process**:
  - Extract text from PDF using libraries like `PyPDF2` or `pdfplumber`.
  - Segment the extracted text into logical chunks (e.g., tables, paragraphs).
  - Convert text chunks into **vector embeddings** using pre-trained embedding models (e.g., OpenAI's `text-embedding-ada-002`).
  - Store the embeddings in a **vector database** (e.g., FAISS, Pinecone).

### Query Handling
- **Input**: User's natural language query.
- **Process**:
  - Convert the query into vector embeddings.
  - Retrieve the most relevant chunks from the vector database based on similarity.

### Comparison Queries
- **Input**: User query for comparing multiple fields across documents.
- **Process**:
  - Identify key comparison terms (e.g., unemployment by degree type).
  - Retrieve relevant data, aggregate it, and format the response in a readable format (e.g., table or bullet points).

### Response Generation
- **Input**: Retrieved data from the vector database and user query.
- **Process**:
  - Use the LLM with augmented retrieval to generate accurate and contextually rich responses.
  - Ensure factuality by incorporating retrieved data directly into the answer.

## Task 2: Chat with Website Using RAG Pipeline

This task involves crawling and scraping structured and unstructured data from websites, converting it into vector embeddings, and enabling users to query the extracted data effectively.

### Data Ingestion
- **Input**: List of URLs to crawl or scrape.
- **Process**:
  - Crawl and scrape content from websites using libraries like `BeautifulSoup` and `Selenium`.
  - Extract key data fields (e.g., text, metadata) and segment the content into chunks.
  - Convert these chunks into **vector embeddings**.
  - Store embeddings in a **vector database** for efficient similarity-based retrieval.

### Query Handling
- **Input**: User's natural language query.
- **Process**:
  - Convert the user's query into vector embeddings.
  - Retrieve the most relevant chunks from the vector database.

### Response Generation
- **Input**: Retrieved data from the vector database and user query.
- **Process**:
  - Generate a detailed response using the LLM, ensuring the response is factually accurate by directly using the retrieved data.

## Installation

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/Prudvi337/sithaphal.git
```

### Dependencies

Make sure you have Python 3.7+ installed. Then, install the required libraries by running:

```bash
pip install -r requirements.txt
```

### Required APIs/Keys
- **Embedding Model**: You will need API access to a pre-trained embedding model (e.g., OpenAI, Hugging Face).
- **Vector Database**: Set up a vector database like FAISS or Pinecone.

## Usage

1. **Ingest Data**:
   - Upload PDF files or provide URLs of websites to scrape.
   
2. **Ask Questions**:
   - Use the provided interface to ask questions based on the ingested data. The system will retrieve relevant chunks and generate responses.

3. **Compare Data**:
   - Ask comparison-based queries, and the system will return aggregated and structured responses.

