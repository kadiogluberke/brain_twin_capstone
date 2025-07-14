import os
import uuid 
from typing import List, Dict, Any, Iterator

from smolagents import CodeAgent, Model, Tool
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama
from PyPDF2 import PdfReader
import streamlit as st

# ------------ DATABASE ---------------------------------------------------------------------

# Persistent ChromaDB 
# chroma_client = chromadb.Client(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_db_data" 
# )

chroma_client = chromadb.Client() 
db = chroma_client.get_or_create_collection("notes")

# ---------------------------------------------------------------------------------------------


# ------------ Embedding & LLM MODEL ---------------------------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Custom Response for LLM ---
class Resp:
    """
    A custom response object to wrap LLM output,
    compatible with smolagents' expectations and
    including token usage information.
    """
    def __init__(self, content_text: str, token_usage_info: Dict[str, Any] = None):
        """
        Initializes the Resp object.

        Args:
            content_text (str): The text content (a chunk for streaming, full text for non-streaming).
            token_usage_info (Dict[str, Any], optional): Dictionary containing token usage details.
                                                         Defaults to an empty dictionary.
                                                         Expected keys: 'prompt_tokens', 'output_tokens', 'total_tokens'.
        """
        self._content = content_text
        self._token_usage = token_usage_info if token_usage_info is not None else {}

    @property
    def content(self) -> str:
        """Returns the text content of the response."""
        return self._content

    @property
    def token_usage(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing token usage information.
        Expected keys: 'prompt_tokens', 'output_tokens', 'total_tokens'.
        """
        return self._token_usage

# --- Custom Llama.cpp Model Wrapper ---
class LlamaCppModel(Model):
    """
    Custom Model wrapper for Llama.cpp (llama-cpp-python library),
    designed to be compatible with smolagents.
    Supports both streaming (`generate_stream`) and non-streaming (`generate`)
    and provides token usage information.
    """
    def __init__(self, path: str, n_gpu_layers: int = 1, n_ctx: int = 4096):
        """
        Initializes the Llama.cpp model.

        Args:
            path (str): Path to the GGUF model file (e.g., "./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf").
            n_gpu_layers (int): Number of layers to offload to GPU (-1 for all, 0 for CPU only).
                                Set to -1 for maximum GPU utilization if a GPU is available.
            n_ctx (int): Context window size for the LLM. This defines how much
                         previous conversation/context the model can "remember".
        """
        print(f"Loading Llama.cpp model from: {path} with n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")
        try:
            self.llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False 
            )
            print("Llama.cpp model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading Llama.cpp model: {e}. Please ensure the model path is correct "
                     "and llama-cpp-python is installed with correct GPU support if n_gpu_layers > 0.")
            self.llm = None

    def _format_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Helper to format messages into a simple prompt string for Llama.cpp."""
        prompt = ""
        for m in messages:
            if m['role'] == 'user':
                prompt += f"USER: {m['content']}\n"
            elif m['role'] == 'assistant':
                prompt += f"ASSISTANT: {m['content']}\n"
            elif m['role'] == 'system':
                prompt += f"SYSTEM: {m['content']}\n"
        prompt += "ASSISTANT:"
        return prompt

    def generate(self, messages: List[Dict], stop_sequences: List[str] = None) -> Resp:
        """
        Generates a non-streaming response from the Llama model.
        This method is called by smolagents for non-streaming operations.

        Args:
            messages (List[Dict]): A list of message dictionaries.
            stop_sequences (List[str]): Optional list of stop sequences.

        Returns:
            Resp: A Resp object containing the full content and total token usage.
        """
        if self.llm is None:
            return Resp(content_text="Error: LLM model not loaded.", token_usage_info={'output_tokens': 0})

        prompt = self._format_messages_to_prompt(messages)

        try:
            out = self.llm(
                prompt,
                max_tokens=512, 
                stop=stop_sequences
            )
            content = out['choices'][0]['text']
            token_usage = {
                'prompt_tokens': out['usage']['prompt_tokens'],
                'output_tokens': out['usage']['completion_tokens'], 
                'total_tokens': out['usage']['total_tokens']
            }
            return Resp(content_text=content, token_usage_info=token_usage)

        except Exception as e:
            print(f"Error during non-streaming LLM generation: {e}")
            return Resp(content_text=f"An error occurred during generation: {e}", token_usage_info={'output_tokens': 0}) 

    def generate_stream(self, messages: List[Dict], stop_sequences: List[str] = None) -> Iterator[Resp]:
        """
        Generates a streaming response from the Llama model.
        This method is called by smolagents when stream_outputs=True.

        Args:
            messages (List[Dict]): A list of message dictionaries.
            stop_sequences (List[str]): Optional list of stop sequences.

        Yields:
            Resp: A Resp object containing the content chunk and token usage for that chunk.
                  Each yielded object represents a piece of the streamed output.
        """
        if self.llm is None:
            yield Resp(content_text="Error: LLM model not loaded.", token_usage_info={'output_tokens': 0}) 
            return

        prompt = self._format_messages_to_prompt(messages)

        try:
            stream_iterator = self.llm(
                prompt,
                max_tokens=512,
                stream=True,
                stop=stop_sequences
            )

            for chunk in stream_iterator:
                text_chunk = chunk['choices'][0]['text']
                token_usage_info = {'output_tokens': 1}
                yield Resp(content_text=text_chunk, token_usage_info=token_usage_info)

        except Exception as e:
            print(f"Error during streaming LLM generation: {e}")
            yield Resp(content_text=f"An error occurred during streaming generation: {e}", token_usage_info={'output_tokens': 0})

# --- Initialize the Llama.cpp Model ---
model = LlamaCppModel("./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", n_gpu_layers=1)


# ---------------------------------------------------------------------------------------------


# ------------ TOOLS ---------------------------------------------------------------------

class UploadNotesTool(Tool):
    name = "upload_notes"
    description = "Upload notes (plain text, markdown, or PDF content) to the knowledge base. This tool processes the content and stores it for later retrieval."
    inputs = {'content': {'type': 'string', 'description': 'The full content of the document to be uploaded.'}}
    output_type = "string"

    def forward(self, content: str) -> str:
        """
        Processes the input content, chunks it, embeds it, and stores it in ChromaDB.
        Generates unique IDs for each chunk using UUIDs.
        """
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
            return f"Successfully uploaded {len(chunks)} chunks of notes. Total notes in DB: {db.count()}."
        except Exception as e:
            return f"Failed to upload notes to ChromaDB: {e}"

class SearchNotesTool(Tool):
    name = "search_notes"
    description = "Search the user's personal notes for relevant information. Use this when the user's question clearly requires looking up details from their uploaded notes. Input should be a concise query or keywords related to the information needed."
    inputs = {'query': {'type': 'string', 'description': 'The specific query or keywords to search within the notes.'}}
    output_type = "string"

    def forward(self, query: str) -> str:
        """
        Searches the ChromaDB vector store for notes relevant to the query.
        Retrieves the top 5 most similar documents.
        """
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

# --- Initialize Tools ---
upload_notes_tool = UploadNotesTool()
search_notes_tool = SearchNotesTool()

# ---------------------------------------------------------------------------------------------


# ------------ AGENT ---------------------------------------------------------------------

agent = CodeAgent(
    tools=[search_notes_tool],
    model=model,
    stream_outputs=True,
    max_steps=3,
    verbosity_level=2,
)

# ---------------------------------------------------------------------------------------------


# ------------ Streamlit ---------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Agentic RAG Notes Assistant")

    st.title("📄🧠 Agentic RAG Notes Assistant")
    st.markdown("Upload your notes and then ask questions. The AI agent will use your notes to answer.")

    with st.sidebar:
        st.header("Upload Your Notes")
        st.info("Supported formats: PDF, Markdown, Plain Text. Notes are stored in an in-memory database for this demo. For persistence, uncomment the `persist_directory` in `chromadb.Client`.")
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
    query = st.text_input("Type your question here:", placeholder="e.g., What is Max's favorite activity? When is the Project Alpha meeting?")

    if query:
        st.markdown("### Agent's Answer:")
        answer_placeholder = st.empty()
        full_response_content = ""
        total_tokens_generated = 0
        

        with st.spinner("Agent is thinking..."):
            try:
                for resp_obj in agent.run(query):
                    if hasattr(resp_obj, 'content'):
                        full_response_content += resp_obj.content
                        answer_placeholder.markdown(full_response_content + "▌")
                    
                    if hasattr(resp_obj, 'token_usage') and isinstance(resp_obj.token_usage, dict):
                        total_tokens_generated += resp_obj.token_usage.get('output_tokens', 0)
                
                answer_placeholder.markdown(full_response_content)
                st.info(f"Total tokens generated: {total_tokens_generated} (approx.)")

            except Exception as e:
                st.error(f"An error occurred while the agent was processing: {e}")
                st.warning("Please check the console for more details (verbosity_level=1 or 2 in CodeAgent).")

    st.sidebar.markdown(f"---")
    st.sidebar.info(f"Notes in DB: {db.count()} chunks")

if __name__ == "__main__":
    main()
