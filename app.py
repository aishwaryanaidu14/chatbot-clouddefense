import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

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

    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/support/solutions/articles/' in href:
            full_url = base_url + href
            links.add(full_url)
        elif '/support/solutions/folders/' in href:
            folder_url = base_url + href
            links.update(fetch_article_links(folder_url, visited_urls))

    return links

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
for url in article_links:
    content = fetch_article_content(url)
    document_chunks[url] = chunk_text(content)

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

for doc_url, chunks in document_chunks.items():
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        embeddings.append(embedding)
        mapping.append((doc_url, i))

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
    ranked_links = [(mapping[i][0], mapping[i][1], similarities[0][i]) for i in sorted_indices[:top_n]]
    return ranked_links
'''def retrieve_document_links(query, embeddings, mapping, model, tokenizer, top_n=3):
    try:
        query_embedding = embed_text(query, model, tokenizer)
        similarities = cosine_similarity([query_embedding], embeddings)
        sorted_indices = np.argsort(similarities[0])[::-1]
        ranked_links = [(mapping[i][0], mapping[i][1], similarities[0][i]) for i in sorted_indices[:top_n]]
        return ranked_links
    except Exception as e:
        print(f"Error retrieving document links: {e}")
        return []

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fetch article links and content
solutions_page = base_url + "/support/solutions"
article_links = fetch_article_links(solutions_page)
article_contents = [fetch_article_content(link) for link in article_links]

# Chunk article content
document_chunks = {url: chunk_text(content) for url, content in zip(article_links, article_contents)}

# Embed chunks
embeddings = []
mapping = []

for doc_url, chunks in document_chunks.items():
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk, model, tokenizer)
        embeddings.append(embedding)
        mapping.append((doc_url, i))

embeddings = np.array(embeddings)

# Save embeddings and mapping
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

with open("mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)
'''
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form.get('query')

        if query is None:
            return jsonify([])
        
        ranked_links = retrieve_document_links(query)
        print("Top ranked links:")
        for link, chunk_idx, score in ranked_links:
            print(f"Link: {link}, Chunk Index: {chunk_idx}, Score: {score}")

        #with open("embeddings.pkl", "rb") as f:
         #   embeddings = pickle.load(f)

        #with open("mapping.pkl", "rb") as f:
          #  mapping = pickle.load(f)

        #ranked_links = retrieve_document_links(query, embeddings, mapping, model, tokenizer)
        return jsonify(ranked_links)
    except Exception as e:
        print(f"Error in search: {e}")
        return jsonify([])
    '''
    query = request.form['query']

    article_links = fetch_article_links(base_url + "/support/solutions")
    article_contents = [fetch_article_content(link) for link in article_links]
    article_chunks = [chunk_text(content) for content in article_contents]
    document_chunks = {link: chunk_text(content) for link, content in zip(article_links, article_contents)}

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    mapping = []

    for doc_url, chunks in document_chunks.items():
        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk, model, tokenizer)
            embeddings.append(embedding)
            mapping.append((doc_url, i))

    embeddings = np.array(embeddings)

    query_embedding = embed_text(query, model, tokenizer)
    similarities = cosine_similarity([query_embedding], embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]
    top_n = 3
    ranked_links = [(mapping[i][0], mapping[i][1], similarities[0][i]) for i in sorted_indices[:top_n]]

    return jsonify(ranked_links)
    '''

if __name__ == '__main__':
    app.run(debug=True)




''' For openai api key
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
from transformers import pipeline

app = Flask(__name__)

openai_pipeline = pipeline("text-generation", model="openai/gpt-3.5-turbo", api_key="OPENAI_API_KEY")

def fetch_article_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    base_url = "https://support.clouddefense.ai"
    article_links = [base_url + link['href'] for link in soup.find_all('a', href=True) if '/support/solutions/articles/' in link['href']]
    return article_links

def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(separator=' ')

def chunk_text(text, max_length=512):
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    return chunks

def embed_text(text):
    return openai_pipeline(text)

def retrieve_document_links(query, top_n=3):
    article_links = fetch_article_links("https://support.clouddefense.ai/support/solutions")
    article_contents = [fetch_article_content(link) for link in article_links]
    article_chunks = [chunk_text(content) for content in article_contents]

    embeddings = [openai_pipeline(chunk) for chunk_list in article_chunks for chunk in chunk_list]

    similarities = [openai_pipeline(f"{query}\n{embedding[0]['generated_text']}") for embedding in embeddings]

    ranked_links = sorted(zip(article_links, similarities), key=lambda x: x[1][0]['score'], reverse=True)[:top_n]
    return [{"url": link[0], "similarity": similarity[0]['score']} for link, similarity in ranked_links]

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if query:
        document_links = retrieve_document_links(query)
        return jsonify(document_links)
    else:
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True)
'''
