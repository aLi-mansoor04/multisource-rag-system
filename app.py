import streamlit as st
import os
import tempfile
from urllib.parse import urlparse, parse_qs

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Bot Hub",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { max-width: 760px; margin: auto; }
    .bot-title { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
    .bot-sub   { color: #888; font-size: 14px; margin-bottom: 24px; }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #4f8ef7;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 16px;
        font-size: 15px;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedding():

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HF_API_KEY"],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

def get_prompt():
    return PromptTemplate(
        template="""
You are a helpful assistant.
Answer only from the provided context and always answer in English even if the input is in Hindi or any other language.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"]
    )

def build_chain(retriever):
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    return parallel | get_prompt() | get_llm() | StrOutputParser()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="bot-title">🤖 AI Bot Hub</p>', unsafe_allow_html=True)
st.markdown('<p class="bot-sub">Choose a bot, provide your source, and ask anything.</p>', unsafe_allow_html=True)

bot = st.selectbox(
    "Select bot",
    ["📄 PDF Bot", "▶️ YouTube Bot (Robust)", "🎬 YouTube Bot (Simple)", "🌐 Website Bot"],
    label_visibility="collapsed"
)

st.divider()


# ── PDF Bot ───────────────────────────────────────────────────────────────────

if bot == "📄 PDF Bot":
    st.subheader("📄 PDF Bot")
    st.caption("Upload a PDF and ask questions about its content.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_input("Your question", placeholder="e.g. What is the main topic?")

    if st.button("Ask", type="primary", disabled=not (uploaded_file and question)):
        with st.spinner("Reading PDF and searching for answer..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.unlink(tmp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(chunks, get_embedding())
            retriever = vectorstore.as_retriever()

            chain = build_chain(retriever)
            answer = chain.invoke(question)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)


# ── YouTube Bot (Robust) ──────────────────────────────────────────────────────

elif bot == "▶️ YouTube Bot (Robust)":
    st.subheader("▶️ YouTube Bot — Robust")
    st.caption("Tries YouTubeTranscriptApi first, falls back to yt-dlp if needed.")

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    question = st.text_input("Your question", placeholder="e.g. What is this video about?")

    if st.button("Ask", type="primary", disabled=not (url and question)):
        with st.spinner("Fetching transcript..."):
            try:
                video_id = parse_qs(urlparse(url).query)["v"][0]
                ytt_api = YouTubeTranscriptApi()
                try:
                    tl = ytt_api.fetch(video_id, languages=["hi", "en"])
                except Exception:
                    tl = ytt_api.fetch(video_id)
                transcript = " ".join(chunk.text for chunk in tl)

            except Exception as e:
                st.warning(f"YouTubeTranscriptApi failed ({e}), trying yt-dlp...")
                try:
                    with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                    raw = info.get("subtitles") or info.get("automatic_captions")
                    if not raw:
                        st.error("No subtitles found for this video.")
                        st.stop()
                    lang_data = raw.get("hi") or raw.get("en") or next(iter(raw.values()))
                    fmt = next((f for f in lang_data if f.get("ext") == "json3"), lang_data[0])
                    import requests
                    caption_json = requests.get(fmt["url"]).json()
                    transcript = " ".join(
                        ev["segs"][0]["utf8"]
                        for ev in caption_json.get("events", [])
                        if "segs" in ev and ev["segs"][0].get("utf8", "\n") != "\n"
                    )
                except Exception as e2:
                    st.error(f"yt-dlp also failed: {e2}")
                    st.stop()

        with st.spinner("Building index and answering..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            vectorstore = FAISS.from_documents(chunks, get_embedding())
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            chain = build_chain(retriever)
            answer = chain.invoke(question)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)


# ── YouTube Bot (Simple) ──────────────────────────────────────────────────────

elif bot == "🎬 YouTube Bot (Simple)":
    st.subheader("🎬 YouTube Bot — Simple")
    st.caption("Uses YouTubeTranscriptApi only. Works best for videos with English/Hindi captions.")

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    question = st.text_input("Your question", placeholder="e.g. Summarize this video")

    if st.button("Ask", type="primary", disabled=not (url and question)):
        with st.spinner("Fetching transcript..."):
            try:
                video_id = parse_qs(urlparse(url).query)["v"][0]
                ytt_api = YouTubeTranscriptApi()
                try:
                    tl = ytt_api.fetch(video_id, languages=["hi", "en"])
                except Exception:
                    tl = ytt_api.fetch(video_id)
                transcript = " ".join(chunk.text for chunk in tl)
            except Exception as e:
                st.error(f"Could not fetch transcript: {e}")
                st.stop()

        with st.spinner("Building index and answering..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            vectorstore = FAISS.from_documents(chunks, get_embedding())
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            chain = build_chain(retriever)
            answer = chain.invoke(question)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)


# ── Website Bot ───────────────────────────────────────────────────────────────

elif bot == "🌐 Website Bot":
    st.subheader("🌐 Website Bot")
    st.caption("Paste any public URL and ask questions about the page content.")

    url = st.text_input("Website URL", placeholder="https://example.com")
    question = st.text_input("Your question", placeholder="e.g. What does this company do?")

    if st.button("Ask", type="primary", disabled=not (url and question)):
        with st.spinner("Loading website..."):
            try:
                loader = WebBaseLoader(web_paths=[url])
                docs = loader.load()
            except Exception as e:
                st.error(f"Could not load URL: {e}")
                st.stop()

        with st.spinner("Building index and answering..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, get_embedding())
            retriever = vectorstore.as_retriever()
            chain = build_chain(retriever)
            answer = chain.invoke(question)

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
