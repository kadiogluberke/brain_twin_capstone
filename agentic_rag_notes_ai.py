from smolagents import CodeAgent, TransformersModel, Tool, Model
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama
from PyPDF2 import PdfReader
import streamlit as st

# ------------ DATABASE ---------------------------------------------------------------------

# Persistent ChromaDB
# chroma_client = chromadb.Client(
#     chromadb.config.Settings(
#         chroma_db_impl="duckdb+parquet",
#         persist_directory="./chroma_db"
#     )
# )
# db = chroma_client.get_or_create_collection("notes")

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection("notes")

# ---------------------------------------------------------------------------------------------


# ------------ Embedding & LLM MODEL ---------------------------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# llm_backend = Llama(model_path="./llama-2-7b.Q2_K.gguf")
# model = TransformersModel.from_callable(llm_backend)

# model = TransformersModel(
#     model_id="meta-llama/Llama-2-7b-chat-hf",
#     device_map="auto",
#     torch_dtype="auto",
#     trust_remote_code=False
# )


class LlamaCppModel(Model):
    def __init__(self, path):
        self.llm = Llama(model_path=path, n_ctx=4096)

    def generate(self, messages, stop_sequences=None):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        out = self.llm(prompt, max_tokens=512)
        class Resp: content = out['choices'][0]['text']
        return Resp()

model = LlamaCppModel("./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")

# ---------------------------------------------------------------------------------------------


# ------------ TOOLS ---------------------------------------------------------------------


class UploadNotesTool(Tool):
    name = "upload_notes"
    description = "Upload notes (plain text)."
    inputs = {'content': {'type': 'string', 'description': 'The content of the document'}}
    output_type = "string"
    def forward(self, content: str) -> str:
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        embs = embed_model.encode(chunks)
        for i, c in enumerate(chunks):
            db.add(documents=[c], embeddings=[embs[i]], ids=[f"note-{i}"])
        return f"Uploaded {len(chunks)} chunks"

class SearchNotesTool(Tool):
    name = "search_notes"
    description = "Search notes"
    inputs = {'query': {'type': 'string', 'description': 'The query from user'}}
    output_type = "string"
    def forward(self, query: str) -> str:
        vec = embed_model.encode([query])[0]
        res = db.query(query_embeddings=[vec], n_results=5)
        return "\n".join(res["documents"][0])
    
# upload_notes_tool = UploadNotesTool()
search_notes_tool = SearchNotesTool()

# ---------------------------------------------------------------------------------------------


# ------------ AGENT ---------------------------------------------------------------------


agent = CodeAgent(
    tools=[search_notes_tool],
    model=model,
    stream_outputs=False,
    max_steps=2,
    verbosity_level=1,
)

# ---------------------------------------------------------------------------------------------


# ------------ Streamlit APP ---------------------------------------------------------------------


def main():
    st.title("📄🧠 Agentic RAG Notes Assistant")
    st.sidebar.header("Upload")
    f = st.sidebar.file_uploader("PDF / MD / TXT", type=["pdf","md","txt"])
    if f:
        txt = PdfReader(f).pages and "\n".join(p.extract_text() or "" for p in PdfReader(f).pages) if f.type=="application/pdf" else f.read().decode("utf-8")
        st.sidebar.write(UploadNotesTool().forward(txt))

    query = st.text_input("Ask a question")
    if query:
        result = agent.run(query)
        st.markdown("### Answer:")
        st.write(result)

if __name__ == "__main__":
    main()
