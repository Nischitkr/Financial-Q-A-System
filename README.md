# Financial Q&A System

This project is a sophisticated AI-powered agent designed to answer complex financial questions about public companies using their 10-K filings. It implements a full Retrieval-Augmented Generation (RAG) pipeline.

This system was built to fulfill the **"AI Engineering Assignment: RAG Sprint Challenge"**.

---

## Features

- **Automated Data Acquisition**: Downloads the latest 10-K filings for Google (GOOGL), Microsoft (MSFT), and NVIDIA (NVDA) directly from the SEC EDGAR database.
- **Robust RAG Pipeline**: Parses HTML filings, splits them into semantic chunks, generates embeddings, and stores them in a local ChromaDB vector store.
- **Intelligent Agent Architecture**:
  - **Query Classifier**: Intelligently determines if a user's question requires a numerical answer ("quantitative") or a descriptive summary ("qualitative").
  - **Query Decomposer**: Breaks down complex, multi-company questions into a series of specific, targeted sub-queries.
  - **Multi-Tool Execution**: Employs two specialized tools—a Number Extractor for quantitative questions and a Text Summarizer for qualitative ones.
  - **Multi-Step Reasoning**: The agent methodically processes each sub-query, retrieves relevant context, filters it for quality, and uses the correct tool to process the information.
  - **Cited and Verifiable Answers**: All answers are returned in a clean JSON format, complete with excerpts and estimated page numbers from the source documents to ensure verifiability.

---

## Project Structure

```
├── main.py             # The main Python script containing all the agent logic.
├── requirements.txt    # A list of all necessary Python libraries.
├── README.md           # This documentation file.
├── data/               # (Auto-generated) Stores the downloaded 10-K filings.
└── vector_store/       # (Auto-generated) Stores the ChromaDB vector database.
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or newer
- `pip` for installing Python packages

### 2. Clone the Repository

Clone this project to your local machine.

### 3. Install Dependencies

Navigate to the project directory in your terminal and install all the required libraries from the requirements.txt file:

```sh
pip install -r requirements.txt
```

### 4. Get a Groq API Key

This project uses the Groq API for its high-speed Large Language Model (LLM) inference.

- Go to the [Groq Console](https://console.groq.com/) and sign up for a free account.
- Create a new API key and copy it.
- Set the API key as an environment variable in your terminal. This keeps your key secure and out of the source code.

**On macOS or Linux:**
```sh
export GROQ_API_KEY="your-api-key-here"
```

**On Windows (Command Prompt):**
```cmd
set GROQ_API_KEY="your-api-key-here"
```

> **Note**: You must set this environment variable every time you open a new terminal session.

---

## How to Use the System

The system is operated via the command line and has two main commands: `build` and `query`.

### Step 1: Build the Knowledge Base

Before you can ask any questions, you must first download the 10-K filings and build the vector database. This command will create the `data/` and `vector_store/` directories.

```sh
python main.py build
```

This process will take a few minutes as it downloads the filings and generates embeddings for all the text. You only need to run this command once (or whenever you want to update the data).

### Step 2: Ask a Question

Once the build is complete, you can ask questions using the `query` command. The question should be enclosed in double quotes.

```sh
python main.py query "Your financial question here"
```

The agent will then execute its multi-step process and print the final, formatted JSON answer to the console.

**Example:**

```sh
python main.py query "How did NVIDIA’s data center revenue grow from 2022 to 2023?"
```

```sh
python main.py query "What percentage of Google’s revenue came from advertising in 2023?"
```
