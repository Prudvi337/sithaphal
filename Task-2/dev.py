import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

class WebScraper:
    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)

    def scrape(self, url, retry_count=3):
        for _ in range(retry_count):
            try:
                self.driver.get(url)
                time.sleep(3)
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                return {
                    "title": soup.title.string if soup.title else "Untitled",
                    "text": ' '.join([p.get_text() for p in soup.find_all('p')]),
                    "links": [a['href'] for a in soup.find_all('a', href=True)]
                }
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                time.sleep(2)
        return None

    def close(self):
        self.driver.quit()

class TextProcessor:
    @staticmethod
    def chunk_text(text, chunk_size=500, overlap=100):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class SemanticSearch:
    def __init__(self, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.metadata = []

    def index_data(self, texts, metadata):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({"metadata": self.metadata[idx], "distance": dist})
        return results

class LLMResponder:
    def __init__(self, model_name='gpt2'):
        self.pipeline = pipeline("text-generation", model=model_name)

    def generate_response(self, context, query, max_length=150):
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        return self.pipeline(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']

class RAGPipeline:
    def __init__(self):
        self.scraper = WebScraper()
        self.text_processor = TextProcessor()
        self.semantic_search = SemanticSearch()
        self.llm_responder = LLMResponder()

    def ingest(self, urls, chunk_size=500, overlap=100):
        all_chunks = []
        metadata = []
        for url in urls:
            scraped_data = self.scraper.scrape(url)
            if scraped_data:
                chunks = self.text_processor.chunk_text(scraped_data['text'], chunk_size, overlap)
                all_chunks.extend(chunks)
                metadata.extend([{"url": url, "title": scraped_data['title']}] * len(chunks))
        self.semantic_search.index_data(all_chunks, metadata)

    def query(self, query_text, top_k=5):
        search_results = self.semantic_search.search(query_text, top_k)
        context = '\n'.join([result['metadata']['url'] + ': ' + result['metadata']['title'] for result in search_results])
        return self.llm_responder.generate_response(context, query_text)

    def close(self):
        self.scraper.close()

if __name__ == "__main__":
    pipeline = RAGPipeline()

    # URLs to ingest
    urls = [
        "https://www.stanford.edu/",
        "https://www.uchicago.edu/", 
        "https://www.washington.edu/",
        "https://www.und.edu/"
    ]

    # Ingest data
    print("Ingesting data...")
    pipeline.ingest(urls)

    # Query the pipeline
    print("Querying pipeline...")
    query_text = "What are the main research areas at Stanford University?"
    response = pipeline.query(query_text)
    print("Response:", response)
    # Cleanup
    pipeline.close()
