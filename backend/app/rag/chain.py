from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient

def get_rag_chain():
    # Load LLM and embedder
    llm = Ollama(model="mistral")
    embeddings = OllamaEmbeddings(model="mistral")

    # Connect to Qdrant using the new API
    client = QdrantClient(url="http://localhost:6333")
    vectorstore = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="docs"
    )

    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
