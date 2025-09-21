import os
import json
import argparse
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# --- CONFIGURATION ---
COMPANY_TICKERS = {"GOOGL": "1652044", "MSFT": "789019", "NVDA": "1045810"}
YEARS = ["2022", "2023", "2024"]
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_CLIENT = Groq()
LLM_MODEL = 'openai/gpt-oss-120b'
# llama-3.3-70b-versatile

# --- DATA ACQUISITION & RAG PIPELINE ---
def get_sec_filings_links_robust(ticker, cik, years):
    links = {}
    headers = {'User-Agent': 'YourName YourEmail@example.com'}
    padded_cik = cik.zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    try:
        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        filings = data['filings']['recent']
        ten_k_filings = {}
        for i in range(len(filings['form'])):
            if filings['form'][i] == '10-K':
                filing_date = filings['filingDate'][i]
                report_year = str(int(filing_date.split('-')[0]))
                if report_year in years and report_year not in ten_k_filings:
                    accession_number = filings['accessionNumber'][i].replace('-', '')
                    primary_document = filings['primaryDocument'][i]
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"
                    ten_k_filings[report_year] = doc_url
        for year in years:
            if year in ten_k_filings:
                links[year] = ten_k_filings[year]
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return {}
    return links

def download_10ks():
    os.makedirs(DATA_DIR, exist_ok=True)
    headers = {'User-Agent': 'YourName YourEmail@example.com'}
    for ticker, cik in COMPANY_TICKERS.items():
        print(f"Fetching 10-K links for {ticker}...")
        links = get_sec_filings_links_robust(ticker, cik, YEARS)
        if not links:
            print(f"Could not find required 10-K links for {ticker}. Skipping.")
            continue
        for year, url in links.items():
            filepath = os.path.join(DATA_DIR, f"{ticker}_{year}_10K.html")
            if os.path.exists(filepath):
                print(f"File already exists: {filepath}. Skipping download.")
                continue
            print(f"Downloading {ticker} {year} 10-K...")
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Saved to {filepath}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")

def build_vector_store():
    print("Initializing RAG pipeline...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection_name = "financial_rag_html"
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
    collection = client.create_collection(name=collection_name)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".html")]
    if not files:
        print("No HTML files found.")
        return
    print("Processing downloaded 10-K filings...")
    doc_id_counter = 0
    for filename in tqdm(files, desc="Processing Files"):
        filepath = os.path.join(DATA_DIR, filename)
        parts = filename.replace("_10K.html", "").split("_")
        ticker, year = parts[0], parts[1]
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = ' '.join(soup.get_text().split())
        chunks = text_splitter.split_text(text)
        metadatas = []
        for i, chunk in enumerate(chunks):
            page_estimate = (i // 2) + 1
            metadatas.append({
                "company": ticker, "year": year, 
                "source": filename, "page": page_estimate
            })
        embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"doc_{doc_id_counter+i}" for i in range(len(chunks))]
        doc_id_counter += len(chunks)
        collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
    print(f"Vector store built successfully with {collection.count()} documents.")


# --- STEP 3: QUERY ENGINE  ---
class FinancialAgent:
    def __init__(self):
        print("Initializing Financial Agent...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        try:
            self.collection = client.get_collection(name="financial_rag_html")
            print("Successfully connected to vector store.")
        except Exception:
            self.collection = None

    def _classify_query(self, query):
        prompt = f"""
        Classify the user query as "quantitative" or "qualitative".
        - "quantitative" asks for numbers (e.g., "What was the revenue?", "Compare margins").
        - "qualitative" asks for descriptions (e.g., "Compare AI investments", "What are the risks?").
        Query: "{query}"
        Return JSON with one key, "type", set to "quantitative" or "qualitative".
        """
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0)
            result = json.loads(response.choices[0].message.content)
            return result.get("type", "quantitative")
        except Exception:
            return "quantitative"

    def _decompose_query(self, query):
        prompt = f"""
        Decompose the query into sub-queries for Microsoft, Google, and NVIDIA if it's comparative.
        Query: "{query}"
        Return JSON with the key "sub_queries".
        """
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0)
            result = json.loads(response.choices[0].message.content)
            return result.get("sub_queries", [query])
        except Exception as e:
            return [query]

    def _extract_data_point(self, sub_query, context_str):
        prompt = f"""
        You are a data extraction bot. Find the answer to "{sub_query}" from the text below.
        Instructions:
        1. Find the exact figure for the metric. If needed, calculate margin = (Operating Income / Revenue) * 100.
        2. "excerpt" MUST be the sentence/data with the value.
        3. "page" MUST be the "Source Page" number.
        4. Return JSON with "value" (float), "excerpt" (string), and "page" (int). If not found, "value" is null.
        Text: <context>{context_str}</context>
        """
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"    -> Extraction API call failed: {e}")
            return {"value": None, "excerpt": "Extraction failed due to API error.", "page": None}

    def _summarize_topic(self, sub_query, context_str):
        prompt = f"""
        You are a financial analyst. Summarize the answer to "{sub_query}" based ONLY on the text below.
        Instructions:
        1. "summary" should be a few sentences.
        2. "excerpt" MUST be a direct quote supporting the summary.
        3. "page" MUST be the "Source Page" number.
        4. Return JSON with "summary" (string), "excerpt" (string), and "page" (int). If not found, summary states that.
        Text: <context>{context_str}</context>
        """
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"    -> Summarization API call failed: {e}")
            return {"summary": "Summarization failed due to API error.", "excerpt": None, "page": None}

    def _final_synthesis_narrative(self, query, extracted_data, query_type):
        prompt = f"""
        Generate the narrative `answer` and `reasoning` for a financial query based on pre-extracted data.
        Original Query: "{query}"
        Query Type: "{query_type}"
        Extracted Data: {json.dumps(extracted_data, indent=2)}
        Instructions:
        1. If "quantitative", compare the "value" for each company. State the winner and list all values in the `answer`.
        2. If "qualitative", combine the "summary" for each company into a coherent paragraph in the `answer`.
        3. The `reasoning` should briefly explain how you came to the answer.
        Return ONLY a JSON object with two keys: "answer" and "reasoning".
        """
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0)
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"answer": "Failed to generate narrative.", "reasoning": "An error occurred."}

    def answer_query(self, query):
        if not self.collection:
            return {"error": "Vector store not available."}

        print("Step 1: Classifying query intent...")
        query_type = self._classify_query(query)
        print(f"  -> Query type: {query_type}")

        print("Step 2: Decomposing query...")
        sub_queries = self._decompose_query(query)
        print(f"  -> Sub-queries: {sub_queries}")

        print(f"Step 3: Executing '{query_type}' tool for each sub-query...")
        extracted_data = []
        for sub_q in sub_queries:
            company_name = "Unknown"
            if "microsoft" in sub_q.lower(): company_name = "MSFT"
            elif "google" in sub_q.lower(): company_name = "GOOGL"
            elif "nvidia" in sub_q.lower(): company_name = "NVDA"

            year_match = re.search(r'\b(2022|2023|2024)\b', sub_q)
            year = year_match.group(0) if year_match else '2023'
            
            print(f"  -> Processing: '{sub_q}' for {company_name} ({year})")
            
            retrieved_chunks = self.collection.query(
                query_texts=[sub_q], n_results=50, include=["metadatas", "documents"])

            if query_type == "quantitative":
                keywords = ["operating income", "operating margin", "statements of operations", "total revenues", "cloud", "data center", "advertising", "r&d", "research and development"]
            else:
                keywords = ["artificial intelligence", "ai ", "risk", "investment", "strategy"]
            
            filtered_docs, filtered_metas = [], []
            for i, doc in enumerate(retrieved_chunks['documents'][0]):
                if any(keyword in doc.lower() for keyword in keywords):
                    filtered_docs.append(doc)
                    filtered_metas.append(retrieved_chunks['metadatas'][0][i])
            
            print(f"    -> Retrieved {len(retrieved_chunks['documents'][0])} chunks, filtered down to {len(filtered_docs)}.")
            
            # We cap the context to the top 10 most relevant filtered chunks to avoid API errors.
            capped_docs = filtered_docs[:10]
            capped_metas = filtered_metas[:10]

            context_str = "\n\n---\n\n".join([f"Source Page: {meta['page']}\nContent: {doc}" for doc, meta in zip(capped_docs, capped_metas)])

            if query_type == "quantitative":
                data_point = self._extract_data_point(sub_q, context_str)
            else:
                data_point = self._summarize_topic(sub_q, context_str)
            
            data_point['company'] = company_name
            data_point['year'] = year
            extracted_data.append(data_point)
        
        print("  -> Data processing complete.")
        print("Processed Data:", json.dumps(extracted_data, indent=2))

        print("Step 4: Synthesizing final answer...")
        narrative = self._final_synthesis_narrative(query, extracted_data, query_type)
        
        final_answer = {"query": query, **narrative, "sub_queries": sub_queries, "sources": []}
        for item in extracted_data:
            source_item = {k: v for k, v in item.items() if k != 'query'}
            final_answer["sources"].append(source_item)

        print("  -> Synthesis complete.")
        return final_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial RAG Q&A System")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Download HTML 10-K filings and build the vector store.")
    query_parser = subparsers.add_parser("query", help="Ask a question to the financial agent.")
    query_parser.add_argument("question", type=str, help="The question you want to ask.")
    args = parser.parse_args()

    if args.command == "build":
        download_10ks()
        build_vector_store()
    elif args.command == "query":
        if "GROQ_API_KEY" not in os.environ:
            print("Error: GROQ_API_KEY environment variable not set.")
        else:
            agent = FinancialAgent()
            result = agent.answer_query(args.question)
            print("\n--- FINAL ANSWER ---\n")
            print(json.dumps(result, indent=2))