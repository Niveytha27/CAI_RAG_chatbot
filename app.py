import requests
import io
import re
import numpy as np
import faiss
import torch
import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bert_score import score

st.title("Financial Document Q&A Chatbot")

torch.backends.cuda.matmul.allow_tf32 = True  # Ensure tensor cores are used
torch.backends.cuda.enable_flash_sdp(True)   # Enable Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("BAAI/bge-large-en")
    accelerator = Accelerator()
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    model.config.use_sliding_window_attention = False
    model = accelerator.prepare(model)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return embedding_model, tokenizer, generator, accelerator

embedding_model, tokenizer, generator, accelerator = load_models()

def download_pdf(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF from {url}: {e}")
        return None

def extract_text_from_pdf(pdf_bytes):
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def preprocess_text(text):
    """Cleans text while retaining financial symbols and ensuring proper formatting."""
    if not text:
        return ""
    
    # Define allowed financial symbols
    financial_symbols = r"\$\€\₹\£\¥\₩\₽\₮\₦\₲"

    # Allow numbers, letters, spaces, financial symbols, common punctuation (.,%/-)
    text = re.sub(fr"[^\w\s{financial_symbols}.,%/₹$€¥£-]", "", text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_resource
def load_and_index_data(pdf_urls):
    all_data = []
    for url in pdf_urls:
        pdf_bytes = download_pdf(url)
        if pdf_bytes:
            text = extract_text_from_pdf(pdf_bytes)
            if text:
                preprocessed_text = preprocess_text(text)
                all_data.append(preprocessed_text)

    def chunk_text(text, chunk_size=700, overlap_size=150):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length and text[end].isalnum():
                last_space = text.rfind(" ", start, end)
                if last_space != -1:
                    end = last_space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_length:
                break
            overlap_start = max(0, end - overlap_size)
            if overlap_start < end:
                last_overlap_space = text.rfind(" ", 0, overlap_start)
                if last_overlap_space != -1 and last_overlap_space > start:
                    start = last_overlap_space + 1
                else:
                    start = end
            else:
                start = end
        return chunks

    chunks = []
    for data in all_data:
        chunks.extend(chunk_text(data))

    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def bm25_retrieval(query, documents, top_k=3):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return [documents[i] for i in np.argsort(bm25.get_scores(query.split()))[::-1][:top_k]]

def adaptive_retrieval(query, index, chunks, top_k=3, bm25_weight=0.5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, dtype=np.float16)
    _, indices = index.search(query_embedding, top_k)
    vector_results = [chunks[i] for i in indices[0]]
    bm25_results = bm25_retrieval(query, chunks, top_k)
    return list(set(vector_results + bm25_results))

def rerank(query, results):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    result_embeddings = embedding_model.encode(results, convert_to_numpy=True)
    similarities = np.dot(result_embeddings, query_embedding.T).flatten()
    return [results[i] for i in np.argsort(similarities)[::-1]], similarities

def merge_chunks(retrieved_chunks, overlap_size=100):
    merged_chunks = []
    buffer = retrieved_chunks[0] if retrieved_chunks else ""
    for i in range(1, len(retrieved_chunks)):
        chunk = retrieved_chunks[i]
        overlap_start = buffer[-overlap_size:]
        overlap_index = chunk.find(overlap_start)
        if overlap_index != -1:
            buffer += chunk[overlap_index + overlap_size:]
        else:
            merged_chunks.append(buffer)
            buffer = chunk
    if buffer:
        merged_chunks.append(buffer)
    return merged_chunks

def calculate_confidence(query, answer):
    P, R, F1 = score([answer], [query], lang="en", verbose=False)
    return F1.item()

def generate_response(query, context):
    prompt = f"""Your task is to analyze the given Context and answer the Question concisely in plain English. 
    **Guidelines:**
    - Do NOT include </think> tag, just provide the final answer only.
    - Provide a direct, factual answer based strictly on the Context.
    - Avoid generating Python code, solutions, or any irrelevant information.
    Context: {context}
    Question: {query}     
    Answer:
    """
    response = generator(prompt, max_new_tokens=150, num_return_sequences=1)[0]['generated_text']
    answer = response.split("Answer:")[1].strip()
    return answer

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pdf_urls = st.text_area("Enter PDF URLs (one per line):", "")
pdf_urls = [url.strip() for url in pdf_urls.split("\n") if url.strip()]

if st.button("Load and Index PDFs"):
    with st.spinner("Loading and indexing PDFs..."):
        index, chunks = load_and_index_data(pdf_urls)
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success("PDFs loaded and indexed successfully.")

if "index" in st.session_state and "chunks" in st.session_state:
    if prompt := st.chat_input("Enter your financial question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            retrieved_chunks = adaptive_retrieval(prompt, st.session_state.index, st.session_state.chunks)
            merged_chunks = merge_chunks(retrieved_chunks, 150)
            reranked_chunks, similarities = rerank(prompt, merged_chunks)
            context = " ".join(reranked_chunks[:3])
            answer = generate_response(prompt, context)
            confidence = calculate_confidence(prompt, answer)
            full_response = f"{answer}\n\nConfidence: {confidence:.2f}"
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

accelerator.free_memory()
