import streamlit as st
import os
import numpy as np
from backend import (
    extract_text_from_pdfs, chunk_text, create_embeddings,
    build_faiss_index, search_index, generate_answer
)

st.set_page_config(page_title="StudyMate: AI PDF Q&A", layout="centered")

st.title("📘 StudyMate – AI PDF Q&A")
st.write("📂 Upload *one or multiple PDFs*, and ask questions in any language.")

# ✅ Sidebar for language selection
language = st.sidebar.selectbox(
    "🌍 Choose Answer Language",
    ["en", "hi", "te", "fr", "es", "de"],  # English, Hindi, Telugu, French, Spanish, German
    index=0
)

# ✅ Allow multiple PDFs
uploaded_files = st.file_uploader("📥 Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        path = os.path.join("temp_" + uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(path)

    with st.spinner("📄 Extracting and indexing all PDFs..."):
        raw_text = extract_text_from_pdfs(file_paths)
        chunks = chunk_text(raw_text)
        embeddings = create_embeddings(chunks)
        index = build_faiss_index(np.array(embeddings))

    st.success("✅ All PDFs processed! Ask a question below.")

    # ✅ Chat section
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("💬 Ask a question about the PDFs:")

    if user_question:
        with st.spinner("🔍 Searching & generating answer..."):
            context = search_index(index, user_question, chunks)
            answer = generate_answer(context, user_question, language)

        # ✅ Save Q&A to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": answer})

        st.markdown("### ✅ Answer:")
        st.write(answer)

    # ✅ Show chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"*Q:* {chat['question']}")
            st.markdown(f"*A:* {chat['answer']}")
            st.markdown("---")