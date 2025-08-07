import os
import uuid
from typing import List, Dict, Any, Iterator
from smolagents import OpenAIServerModel
from dotenv import load_dotenv

from smolagents import PythonInterpreterTool, DuckDuckGoSearchTool, VisitWebpageTool
from smolagents import Tool
from sentence_transformers import SentenceTransformer
import chromadb
from PyPDF2 import PdfReader
import streamlit as st
import yaml
from smolagents import ToolCallingAgent

# ------------ DATABASE ---------------------------------------------------------------------

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection("notes")

# ---------------------------------------------------------------------------------------------


# ------------ Embedding & LLM MODEL ---------------------------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=OPENAI_API_KEY,
)
# ---------------------------------------------------------------------------------------------


# ------------ TOOLS ---------------------------------------------------------------------

class UploadNotesTool(Tool):
    name = "upload_notes"
    description = "Upload notes (plain text, markdown, or PDF content) to the knowledge base. This tool processes the content and stores it for later retrieval."
    inputs = {'content': {'type': 'string', 'description': 'The full content of the document to be uploaded.'}}
    output_type = "string"

    def forward(self, content: str) -> str:
        if not content.strip():
            return "Error: No content provided for upload."
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        
        if not chunks:
            return "No processable chunks found in the content."

        embs = embed_model.encode(chunks).tolist()

        batch_id = str(uuid.uuid4())
        ids_to_add = [f"note-{batch_id}-{i}" for i, _ in enumerate(chunks)]

        try:
            db.add(
                documents=chunks,
                embeddings=embs,
                ids=ids_to_add
            )
            return f"Successfully uploaded {len(chunks)} chunks of notes."
        except Exception as e:
            return f"Failed to upload notes to ChromaDB: {e}"

class SearchNotesTool(Tool):
    name = "search_notes"
    description = "Search the user's personal notes for relevant information. Use this when the user's question clearly requires looking up details from their uploaded notes. Input should be a concise query or keywords related to the information needed."
    inputs = {'query': {'type': 'string', 'description': 'The specific query or keywords to search within the notes.'}}
    output_type = "string"

    def forward(self, query: str) -> str:
        if not query.strip():
            return "Please provide a query to search your notes."

        try:
            vec = embed_model.encode([query]).tolist()[0] 
            res = db.query(query_embeddings=[vec], n_results=5, include=['documents'])
            
            if not res['documents'] or not res['documents'][0]:
                return "No relevant information found in your notes for your query."
            retrieved_content = "\n\n".join(res["documents"][0])
            return f"Found the following relevant notes:\n{retrieved_content}"
        except Exception as e:
            return f"An error occurred while searching notes: {e}"

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False


# --- Initialize Tools ---
upload_notes_tool = UploadNotesTool()
search_notes_tool = SearchNotesTool()
final_answer = FinalAnswerTool()

# ---------------------------------------------------------------------------------------------


# ------------ AGENT ---------------------------------------------------------------------

# with open("prompts.yaml", 'r') as stream:
#     prompt_templates = yaml.safe_load(stream)

agent = ToolCallingAgent(
    model=model,
    tools=[
        final_answer,
        search_notes_tool,
        PythonInterpreterTool(),
        # DuckDuckGoSearchTool(),
        # VisitWebpageTool()
    ],
    max_steps=2,
    stream_outputs=True,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    add_base_tools=False 
)

# agent.prompt_templates = prompt_templates

print("Agent successfully configured with the specified tools!")

# ---------------------------------------------------------------------------------------------


# ------------ Streamlit ---------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Agentic RAG Notes Assistant")

    st.title("📄🧠 Agentic RAG Notes Assistant")
    st.markdown("Upload your notes and then ask questions. The AI agent will use your notes to answer.")

    with st.sidebar:
        st.header("Upload Your Notes")
        st.info("Supported formats: PDF, Markdown, Plain Text.")
        f = st.file_uploader("Upload Document", type=["pdf","md","txt"])
        if f:
            with st.spinner("Processing and uploading notes..."):
                txt = ""
                if f.type == "application/pdf":
                    try:
                        reader = PdfReader(f)
                        txt = "\n".join(p.extract_text() or "" for p in reader.pages)
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
                        txt = ""
                else:
                    txt = f.read().decode("utf-8")

                if txt:
                    upload_result = upload_notes_tool.forward(txt)
                    st.success(upload_result)
                else:
                    st.warning("No text extracted from the uploaded file.")

    # Main Chat Interface
    st.header("Ask a Question")
    query = st.text_input("Type your question here:", placeholder="e.g., Ask your Question")

    if query:
        st.markdown("### Agent's Answer:")
        answer_placeholder = st.empty()
        full_response_content = ""
        with st.spinner("Agent is thinking..."):
            try:
                full_response_content = agent.run(query)
                st.markdown(full_response_content)
                

            except Exception as e:
                st.error(f"An error occurred while the agent was processing: {e}")

    st.sidebar.markdown(f"---")
    # st.sidebar.info(f"Notes in DB: {db.count()} chunks")


if __name__ == "__main__":
    main()
