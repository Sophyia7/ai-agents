from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

Settings.llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1, request_timeout=200)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

url = ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
documents = BeautifulSoupWebReader().load_data(url)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

query_engine = index.as_query_engine()
response = query_engine.query("Summarize the document in two paragraphs")
print(response)