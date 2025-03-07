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
import spacy  # for entity extraction  # エンティティ抽出用
import networkx as nx  # for building Knowledge Graph  # ナレッジグラフ構築用
import matplotlib.pyplot as plt  # for visualization  # 可視化用

# ----------------------------------
# Step 1: Load PDF and create documents
# ----------------------------------
loader = PyMuPDFLoader("2501.09223v1.pdf")
documents = loader.load()

# Combine all pages text for KG extraction
full_text = "\n".join([doc.page_content for doc in documents])

# ----------------------------------
# Step 2: Build Knowledge Graph from PDF content using spaCy
# ----------------------------------
# Load spaCy model for NER (choose appropriate model, e.g., en_core_web_sm or a Japanese model)
nlp = spacy.load("en_core_web_sm")  # 英語の場合。日本語の場合は、"ja_core_news_sm"などに変更

doc_spacy = nlp(full_text)
# Extract entities: tuple of (text, label)
entities = [(ent.text, ent.label_) for ent in doc_spacy.ents]

# Create a set of unique entity texts
unique_entities = set(ent[0] for ent in entities)

# Build the Knowledge Graph: nodes are entities, edges represent co-occurrence in the same sentence.
KG = nx.Graph()

# Add nodes to the graph
for entity in unique_entities:
    KG.add_node(entity)

# Add edges between entities co-occurring in the same sentence
for sent in doc_spacy.sents:
    sent_doc = nlp(sent.text)
    sent_entities = [ent.text for ent in sent_doc.ents]
    for i in range(len(sent_entities)):
        for j in range(i + 1, len(sent_entities)):
            if KG.has_edge(sent_entities[i], sent_entities[j]):
                KG[sent_entities[i]][sent_entities[j]]['weight'] += 1
            else:
                KG.add_edge(sent_entities[i], sent_entities[j], weight=1)

# Optional: Save or visualize the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(KG, k=0.5)
nx.draw(KG, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10, edge_color="gray")
plt.title("Knowledge Graph from PDF Document")
plt.savefig("knowledge_graph.png")  # Save the figure if needed
# plt.show()  # Uncomment if you want to display the graph immediately

# ----------------------------------
# Step 3: Split document into chunks for vector search
# ----------------------------------
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

# ----------------------------------
# Step 5: Integrate KG retrieval with vector retrieval
# ----------------------------------
def query_kg(query):
    """Simple KG query: return entities whose name or info contains the query keyword."""
    related_info = []
    for node in KG.nodes():
        if query.lower() in node.lower():
            related_info.append(f"{node}")
    return "\n".join(related_info)

def retrieve_context(question):
    """Combine vector-based context with KG results."""
    vector_results = retriever.invoke(question)
    vector_context = "\n\n".join([doc.page_content for doc in vector_results])
    kg_context = query_kg(question)
    combined_context = vector_context + "\n\nKG Context:\n" + kg_context
    return combined_context

# ----------------------------------
# Step 6: Define LLM query function and Gradio interface
# ----------------------------------
llm = Ollama(model="deepseek-r1")

def query_deepseek(question, context):
    """Query the LLM with the combined context."""
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    print("Formatted Prompt:\n", formatted_prompt)
    response = llm.invoke(formatted_prompt)
    final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return formatted_prompt, final_answer

def ask_question(question):
    """Retrieve context from both vector DB and KG then generate an answer."""
    context = retrieve_context(question)
    full_prompt, answer = query_deepseek(question, context)
    return context, full_prompt, answer

interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs=[
        gr.Textbox(label="Retrieved Context"),
        gr.Textbox(label="Full Prompt Sent to AI"),
        gr.Textbox(label="AI-generated Answer"),
    ],
    title="RAG Chatbot with Knowledge Graph",
    description="Ask any question about the Foundations of LLMs book. Powered by DeepSeek-R1."
)

interface.launch()
