import os
import hashlib
import tempfile
import logging

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("contract_rag")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
FETCH_K = 20
BATCH_SIZE = 100
MAX_HISTORY_TURNS = 4
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION = "contract_collection"

SYSTEM_TEMPLATE = """You are a careful contract and policy review assistant.
Answer ONLY from the numbered context below. Do not make up information.
Rules:
- Cite the supporting passages inline like [1], [2].
- If the context does not contain the answer, reply exactly:
  "I could not find this in the provided documents."
- Quote exact figures, dates, durations and party names as written.
- You provide information from the documents, not legal advice.

Context:
{context}"""

st.set_page_config(page_title="Contract Review Assistant", page_icon="📄", layout="wide")


# cached resources (created once, shared across reruns) 
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


@st.cache_resource
def get_model(api_key: str):
    return init_chat_model(
        "google_genai:gemini-2.5-flash",
        api_key=api_key,
        temperature=0,
        max_retries=3,
        timeout=60,
    )


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    try:
        key = key or st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return key or st.sidebar.text_input("Gemini API key", type="password")


# ingestion 
def make_chunk_id(chunk, idx):
    raw = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{idx}|{chunk.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ingest_uploaded_files(files):
    vector_store = get_vector_store()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total_new = 0
    for f in files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name
            pages = PyPDFLoader(tmp_path).load()
            for p in pages:
                p.metadata["source"] = f.name
            chunks = splitter.split_documents(pages)
            ids = [make_chunk_id(c, i) for i, c in enumerate(chunks)]
            existing = set(vector_store.get(ids=ids)["ids"])
            new = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
            for start in range(0, len(new), BATCH_SIZE):
                batch = new[start:start + BATCH_SIZE]
                vector_store.add_documents(
                    documents=[c for _, c in batch], ids=[i for i, _ in batch]
                )
            total_new += len(new)
        except Exception as e:
            st.sidebar.error(f"Could not process {f.name}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return total_new


def indexed_files():
    metas = get_vector_store().get(include=["metadatas"])["metadatas"]
    return sorted({m.get("source", "unknown") for m in metas})


#  RAG
def retrieve_context(query: str, k: int = TOP_K):
    docs = get_vector_store().max_marginal_relevance_search(query, k=k, fetch_k=FETCH_K)
    context = "\n\n".join(
        f"[{n}] (file: {source_name(d)}, page: {page_number(d)})\n{d.page_content}"
        for n, d in enumerate(docs, start=1)
    )
    return context, docs


def source_name(doc):
    return os.path.basename(str(doc.metadata.get("source", "unknown")))


def page_number(doc):
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else "?"


def condense_question(model, user_query, history):
    if not history:
        return user_query
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-2 * MAX_HISTORY_TURNS:])
    prompt = (
        "Rewrite the last user question as one standalone question, using the "
        "conversation for context. Return ONLY the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {user_query}"
    )
    try:
        return model.invoke([{"role": "user", "content": prompt}]).content.strip()
    except Exception as e:
        log.warning(f"Condense failed, using original query: {e}")
        return user_query


def stream_answer(model, user_query, history):
    """Returns (token generator, source docs). Generator yields text pieces."""
    standalone = condense_question(model, user_query, history)
    context, docs = retrieve_context(standalone)
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
    messages += history[-2 * MAX_HISTORY_TURNS:]
    messages.append({"role": "user", "content": user_query})

    def gen():
        try:
            for chunk in model.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            yield "Sorry, the language model is unavailable right now. Please try again."

    return gen(), docs


def format_sources(docs):
    return [(n, source_name(d), page_number(d), d.page_content)
            for n, d in enumerate(docs, start=1)]


#  UI 
st.title("📄 Contract & Policy Review Assistant")
st.caption("Answers come only from your uploaded documents, with file and page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources"}

with st.sidebar:
    st.header("Documents")
    api_key = get_api_key()
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index documents", disabled=not uploads, use_container_width=True):
        with st.spinner("Reading, chunking and embedding..."):
            added = ingest_uploaded_files(uploads)
        st.success(f"Added {added} new chunks." if added else "Already indexed - nothing new.")

    files = indexed_files()
    st.markdown("**Indexed files**")
    if files:
        for name in files:
            st.write(f"• {name}")
    else:
        st.info("No documents indexed yet.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for n, fname, page, text in m["sources"]:
                    st.markdown(f"**[{n}] {fname} - page {page}**")
                    st.caption(text)

if prompt := st.chat_input("Ask about your contracts or policies..."):
    if not api_key:
        st.error("Add your Gemini API key in the sidebar (or set GEMINI_API_KEY).")
        st.stop()
    if not indexed_files():
        st.error("Upload and index at least one PDF first.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = get_model(api_key)
        generator, docs = stream_answer(model, prompt, history)
        answer = st.write_stream(generator)
        sources = format_sources(docs)
        with st.expander("Sources"):
            for n, fname, page, text in sources:
                st.markdown(f"**[{n}] {fname} - page {page}**")
                st.caption(text)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
