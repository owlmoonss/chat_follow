import re
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from chromadb.config import Settings
from chromadb import Client
from langchain_community.vectorstores import Chroma


# Load the document using PyMuPDFLoader
loader = PyMuPDFLoader("2501.09223v1.pdf")
documents = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Initialize Ollama embeddings using DeepSeek-R1
embedding_function = OllamaEmbeddings(model="deepseek-r1")

# Parallelize embedding generation
def generate_embedding(chunk):
    return embedding_function.embed_query(chunk.page_content)

with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, chunks))

# Initialize Chroma client and create/reset the collection
client = Client(Settings())
collection = client.create_collection(name="foundations_of_llms")

# Add documents and embeddings to Chroma
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content],
        metadatas=[{'id': idx}],
        embeddings=[embeddings[idx]],
        ids=[str(idx)]
    )

# Initialize retriever using Ollama embeddings for queries
retriever = Chroma(
    collection_name="foundations_of_llms",
    client=client,
    embedding_function=embedding_function
).as_retriever()

# Initialize Ollama chat model for DeepSeek-R1
llm = Ollama(model="deepseek-r1")

def retrieve_context(question):
    """Retrieve relevant context from the vector database."""
    results = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

def query_deepseek(question, context):
    """Use DeepSeek-R1 to generate an answer based on the retrieved context."""
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    
    # Print the full prompt for debugging
    print("Formatted Prompt:\n", formatted_prompt)

    # Query DeepSeek-R1 using Ollama
    response = llm.invoke(formatted_prompt)

    # Clean and return the response
    final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return formatted_prompt, final_answer  # Return both prompt and answer

def ask_question(question):
    """Retrieve context and generate an answer using RAG."""
    context = retrieve_context(question)  # Get the retrieved context
    full_prompt, answer = query_deepseek(question, context)  # Get prompt and answer
    
    return context, full_prompt, answer  # Return all for UI display

# Set up the Gradio interface to display context, full prompt, and answer
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs=[
        gr.Textbox(label="Retrieved Context"),  # Show retrieved text from ChromaDB
        gr.Textbox(label="Full Prompt Sent to AI"),  # Show full input prompt
        gr.Textbox(label="AI-generated Answer"),  # Show final AI response
    ],
    title="RAG Chatbot: Foundations of LLMs",
    description="Ask any question about the Foundations of LLMs book. Powered by DeepSeek-R1."
)

interface.launch()
