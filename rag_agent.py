import os
import uuid
from typing import List, Dict, Any, Iterator
from types import SimpleNamespace

from smolagents import FinalAnswerTool, PythonInterpreterTool, DuckDuckGoSearchTool, VisitWebpageTool
from smolagents import CodeAgent, Model, Tool
from smolagents.models import ChatMessageStreamDelta, ChatMessage # Import ChatMessage
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama
from PyPDF2 import PdfReader
import streamlit as st
import yaml

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
    def __init__(self, content_text: str, token_usage_info: Dict[str, Any] = None):

        self._content = content_text
        self._token_usage = {
            'input_tokens': token_usage_info.get('input_tokens', 0) if token_usage_info else 0,
            'output_tokens': token_usage_info.get('output_tokens', 0) if token_usage_info else 0,
            'total_tokens': token_usage_info.get('total_tokens', 0) if token_usage_info else 0,
        }

    @property
    def content(self) -> str:
        return self._content

    @property
    def token_usage(self) -> SimpleNamespace:
        """
        Expected keys: 'input_tokens', 'output_tokens', 'total_tokens'.
        """
        return SimpleNamespace(**self._token_usage)

# --- Custom Llama.cpp Model Wrapper ---
class LlamaCppModel(Model):
    def __init__(self, path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):

        print(f"Loading Llama.cpp model from: {path} with n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")

        try:
            self.llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                chat_format="chatml"
            )
            print("Llama.cpp model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading Llama.cpp model: {e}. Please ensure the model path is correct "
                     "and llama-cpp-python is installed with correct GPU support if n_gpu_layers > 0.")
            self.llm = None

    def _format_messages_for_llamacpp(self, messages: List[Any]) -> List[Dict]:
        formatted_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                formatted_messages.append({'role': m.role, 'content': m.content})
            elif isinstance(m, dict):
                formatted_messages.append(m)
            else:
                print(f"Warning: Unexpected message type encountered: {type(m)}. Attempting to convert.")
                try:
                    formatted_messages.append({'role': m.role, 'content': m.content})
                except AttributeError:
                    print(f"Error: Could not convert message of type {type(m)} to expected format.")
        return formatted_messages


    def generate(self, messages: List[Dict], stop_sequences: List[str] = None) -> Resp:

        if self.llm is None:
            return Resp(content_text="Error: LLM model not loaded.", token_usage_info={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})

        llama_cpp_messages = self._format_messages_for_llamacpp(messages)

        try:
            out = self.llm.create_chat_completion(
                messages=llama_cpp_messages,
                max_tokens=512,
                stop=stop_sequences
            )
            content = out['choices'][0]['message']['content']
            token_usage = {
                'input_tokens': out['usage']['prompt_tokens'],
                'output_tokens': out['usage']['completion_tokens'],
                'total_tokens': out['usage']['total_tokens']
            }
            return Resp(content_text=content, token_usage_info=token_usage)

        except Exception as e:
            print(f"Error during non-streaming LLM generation: {e}")
            return Resp(content_text=f"An error occurred during generation: {e}", token_usage_info={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})

    def generate_stream(self, messages: List[Dict], stop_sequences: List[str] = None) -> Iterator[ChatMessageStreamDelta]:

        if self.llm is None:
            yield ChatMessageStreamDelta(
                content="Error: LLM model not loaded.",
                token_usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)
            )
            return

        llama_cpp_messages = self._format_messages_for_llamacpp(messages)

        try:
            stream_iterator = self.llm.create_chat_completion(
            messages=llama_cpp_messages,
            max_tokens=512,
            stream=True,
            stop=stop_sequences
        )

            initial_input_tokens = 0 

            for chunk in stream_iterator:
                # print(f"DEBUG CHUNK: {chunk}") # For debugging the chunk structure

                delta_content = chunk['choices'][0]['delta'].get('content', '')

                # Only yield if there's actual content to avoid empty deltas
                if delta_content:
                    # Llama.cpp streaming usually has 'usage' in the final chunk,
                    # but sometimes in intermediate. Best to handle robustly.
                    # token_usage_info = {
                    #     'input_tokens': chunk['usage']['prompt_tokens'] if 'usage' in chunk else 0, # Or try to track if initial_input_tokens available
                    #     'output_tokens': chunk['usage']['completion_tokens'] if 'usage' in chunk else 0, # Set to 0 for intermediate, 1 for final chunk token
                    #     'total_tokens': chunk['usage']['total_tokens'] if 'usage' in chunk else 0,
                    # }

                    token_usage_info= {
                        'input_tokens':  0, # Or try to track if initial_input_tokens available
                        'output_tokens':  0, # Set to 0 for intermediate, 1 for final chunk token
                        'total_tokens':  0,
                    }

                    # A more reliable way for streaming might be to provide just the content,
                    # and let smolagents handle token counting or provide a simplified token_usage for deltas.
                    # For simplicity, let's just make sure it's always a SimpleNamespace.
                    yield ChatMessageStreamDelta(
                        content=delta_content,
                        token_usage=SimpleNamespace(**token_usage_info) # Ensure this is always a SimpleNamespace
                    )
                # If the chunk signals end of stream without content, it might contain final usage info
                # You might need to check if chunk.get('choices') and chunk['choices'][0].get('finish_reason') is present
                # to handle the end of stream more precisely.

        except Exception as e:
            print(f"Error during streaming LLM generation: {e}")
            yield ChatMessageStreamDelta(
                content=f"An error occurred during streaming generation: {e}",
                token_usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)
            )

# --- Initialize the Llama.cpp Model ---
model = LlamaCppModel("./qwen2.5-coder-7b-instruct-q4_k_m.gguf", n_gpu_layers=10)


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
            return f"Successfully uploaded {len(chunks)} chunks of notes. Total notes in DB: {db.count()}."
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

# --- Initialize Tools ---
upload_notes_tool = UploadNotesTool()
search_notes_tool = SearchNotesTool()

# ---------------------------------------------------------------------------------------------


# ------------ AGENT ---------------------------------------------------------------------

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


system_prompt_content = prompt_templates['system_prompt']

agent = CodeAgent(
    model=model,
    tools=[
        FinalAnswerTool(),
        search_notes_tool,
        PythonInterpreterTool(),
        DuckDuckGoSearchTool(),
        VisitWebpageTool()
    ],
    max_steps=2,
    stream_outputs=True,
    verbosity_level=2,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    # prompt_templates=prompt_templates,
    # system_prompt=system_prompt_content,
    add_base_tools=False 
)

agent.prompt_templates = prompt_templates

print("Agent successfully configured with the specified tools!")

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
                # for resp_obj in agent.run(query):
                #     if hasattr(resp_obj, 'content'):
                #         full_response_content += resp_obj.content
                #         answer_placeholder.markdown(full_response_content + "▌")
                    
                #     if hasattr(resp_obj, 'token_usage') and isinstance(resp_obj.token_usage, SimpleNamespace):
                #         total_tokens_generated += resp_obj.token_usage.output_tokens
                
                # answer_placeholder.markdown(full_response_content)
                # st.info(f"Total tokens generated: {total_tokens_generated} (approx.)")
                agent.run(query)

            except Exception as e:
                st.error(f"An error occurred while the agent was processing: {e}")
                st.warning("Please check the console for more details (verbosity_level=1 or 2 in CodeAgent).")

    st.sidebar.markdown(f"---")
    st.sidebar.info(f"Notes in DB: {db.count()} chunks")




    # query = "hi, how  are you? What can you do?"
    # res = ""

    # for resp_obj in agent.run(query):
    #     # print("################# RESP OBJ #####################")
    #     res += resp_obj
    #     # print("################# TOKEN USAGE #####################")
    #     # print(resp_obj.token_usage.output_tokens)


    # # print(agent.run(query))
    # print("################# RES #####################")
    # print(res)
    




if __name__ == "__main__":
    main()
