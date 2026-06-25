from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
def main():
    print("Hello from ecommerce-agent!")
    index_name = os.environ.get('PINECONE_INDEX_NAME')
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)


if __name__ == "__main__":
    main()
