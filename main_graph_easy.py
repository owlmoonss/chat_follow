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
import networkx as nx  # For creating a Knowledge Graph

# ----------------------------
# Step 1: Load and process the document
# ----------------------------
# Load the document using PyMuPDFLoader
loader = PyMuPDFLoader("2501.09223v1.pdf")
documents = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# ----------------------------
# Step 2: Generate embeddings and build vector store
# ----------------------------
embedding_function = OllamaEmbeddings(model="deepseek-r1")

def generate_embedding(chunk):
    # Generate an embedding for each text chunk
    return embedding_function.embed_query(chunk.page_content)

with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, chunks))

# Initialize Chroma client and create collection
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

# Set up vector retriever
retriever = Chroma(
    collection_name="foundations_of_llms",
    client=client,
    embedding_function=embedding_function
).as_retriever()

# ----------------------------
# Step 3: Build the Knowledge Graph (using NetworkX)
# ----------------------------
# Here, we simulate entity extraction and KG construction.
# In a real scenario, use NLP methods (e.g., named entity recognition) to extract nodes and edges.
KG = nx.Graph()

# Example: Add nodes and edges manually or through an extraction process
KG.add_node("Foundations", info="Concept related to LLMs foundations")
KG.add_node("DeepSeek-R1", info="LLM model used for querying")
KG.add_edge("Foundations", "DeepSeek-R1", relation="powered_by")

def query_kg(query):
    """Retrieve related entities from the KG based on simple keyword matching."""
    related_info = []
    for node in KG.nodes(data=True):
        if query.lower() in node[0].lower() or query.lower() in node[1].get("info", "").lower():
            related_info.append(f"{node[0]}: {node[1].get('info')}")
    return "\n".join(related_info)

# ----------------------------
# Step 4: Define combined retrieval and query functions
# ----------------------------
# Initialize the LLM for query processing
llm = Ollama(model="deepseek-r1")

def retrieve_context(question):
    """Retrieve context from both vector DB and KG."""
    # Retrieve text chunks from the vector database
    vector_results = retriever.invoke(question)
    vector_context = "\n\n".join([doc.page_content for doc in vector_results])
    
    # Retrieve additional structured context from the KG
    kg_context = query_kg(question)
    
    # Combine both contexts
    combined_context = vector_context + "\n\n" + kg_context
    return combined_context

def query_deepseek(question, context):
    """Generate answer using DeepSeek-R1 with the combined context."""
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    
    # Print the full prompt for debugging
    print("Formatted Prompt:\n", formatted_prompt)
    
    # Query the DeepSeek-R1 model
    response = llm.invoke(formatted_prompt)
    
    # Clean and return the response
    final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return formatted_prompt, final_answer  # Return both prompt and answer

def ask_question(question):
    """Retrieve combined context and generate an answer using RAG with KG."""
    context = retrieve_context(question)  # Get the combined context
    full_prompt, answer = query_deepseek(question, context)  # Get prompt and answer
    return context, full_prompt, answer  # Return all for UI display

# ----------------------------
# Step 5: Set up the Gradio interface
# ----------------------------
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs=[
        gr.Textbox(label="Retrieved Context"),  # Show combined context from ChromaDB and KG
        gr.Textbox(label="Full Prompt Sent to AI"),  # Show full input prompt
        gr.Textbox(label="AI-generated Answer"),  # Show final AI response
    ],
    title="RAG Chatbot with Knowledge Graph",
    description="Ask any question about the Foundations of LLMs book with enhanced KG context. Powered by DeepSeek-R1."
)

interface.launch()
