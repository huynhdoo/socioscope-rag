from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def load_vectorstore(path, api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vector_store = InMemoryVectorStore.load(path, embeddings)
    print(f"Load {len(vector_store.store)} documents.")
    return vector_store