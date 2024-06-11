import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import concurrent.futures

app = Flask(__name__)

base_url = "https://support.clouddefense.ai"

def fetch_article_links(page_url, visited_urls=None):
    if visited_urls is None:
        visited_urls = set()

    if page_url in visited_urls:
        return set()
    
    visited_urls.add(page_url)
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {
            executor.submit(process_link, base_url + link['href'], visited_urls): link['href']
            for link in soup.find_all('a', href=True)
            if '/support/solutions/articles/' in link['href'] or '/support/solutions/folders/' in link['href']
        }
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                result = future.result()
                if isinstance(result, set):
                    links.update(result)
                else:
                    links.add(result)
            except Exception as exc:
                print(f"Error fetching links: {exc}")

    return links

def process_link(href, visited_urls):
    if '/support/solutions/articles/' in href:
        return href
    elif '/support/solutions/folders/' in href:
        return fetch_article_links(href, visited_urls)

solutions_page = base_url + "/support/solutions"
article_links = fetch_article_links(solutions_page)
print(f"Found {len(article_links)} article links.")

def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_content = soup.get_text(separator=' ')
    return article_content.strip()

def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

document_chunks = {}

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_url = {executor.submit(fetch_article_content, url): url for url in article_links}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            content = future.result()
            document_chunks[url] = chunk_text(content)
        except Exception as exc:
            print(f"Error fetching content from {url}: {exc}")

print(f"Chunked {len(document_chunks)} documents.")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

embeddings = []
mapping = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_chunk = {
        executor.submit(embed_text, chunk): (doc_url, i)
        for doc_url, chunks in document_chunks.items()
        for i, chunk in enumerate(chunks)
    }
    for future in concurrent.futures.as_completed(future_to_chunk):
        doc_url, i = future_to_chunk[future]
        try:
            embedding = future.result()
            embeddings.append(embedding)
            mapping.append((doc_url, i))
        except Exception as exc:
            print(f"Error creating embedding for chunk {i} of {doc_url}: {exc}")

embeddings = np.array(embeddings)
print(f"Created embeddings for {len(embeddings)} chunks.")

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

with open("mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

def retrieve_document_links(query, top_n=3):
    query_embedding = embed_text(query)
    similarities = cosine_similarity([query_embedding], embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]
    ranked_links = [(mapping[i][0], mapping[i][1], float(similarities[0][i])) for i in sorted_indices[:top_n]]
    return ranked_links

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_links', methods=['POST'])
def get_links():
    data = request.json
    query = data['query']
    ranked_links = retrieve_document_links(query)
    response = [{'link': link, 'chunk_idx': chunk_idx, 'score': score} for link, chunk_idx, score in ranked_links]
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
