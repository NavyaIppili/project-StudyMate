import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# ✅ Embedding model (fast)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Translate text if needed
def translate_text(text, target_lang="en"):
    if target_lang == "en":
        return text
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# ✅ Extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    """Combine text from multiple PDFs"""
    all_text = ""
    for file_path in pdf_files:
        doc = fitz.open(file_path)
        for page in doc:
            all_text += page.get_text()
    return all_text

# ✅ Chunk text
def chunk_text(text, max_length=400):
    """Split text into chunks for embeddings"""
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ✅ Create embeddings
def create_embeddings(chunks):
    return embedding_model.encode(chunks)

# ✅ Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ✅ Search FAISS index and return top chunks instantly
def search_index(index, query, chunks, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [chunks[i] for i in indices[0]]
    return " ".join(results)

# ✅ Generate answer (FAST – no HuggingFace model)
def generate_answer(context, question, language="en"):
    # Just return the most relevant text chunk as the answer
    answer = f"{context}\n\n(Source: PDF)"
    return translate_text(answer, language)