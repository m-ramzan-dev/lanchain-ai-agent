from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

from pinecone import Pinecone, ServerlessSpec

def main():
    print("ingestion!")
    loader = PyPDFLoader(file_path="/Users/ramzan/Downloads/plum-networks-confidentiality-agreement_muhammad_ramzan.pdf")
    documents = loader.load()
    print("pdf file loaded")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=30
    )

    docs = text_splitter.split_documents(documents=documents)
    print(f"document splited into {docs} chunks")

    embeddings = OpenAIEmbeddings()
    print("ingesting...")
    index_name = os.environ.get('PINECONE_INDEX_NAME')
    
    pc = Pinecone()
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    PineconeVectorStore.from_documents(
        docs, embeddings,index_name=index_name
    )
    

    print("finish")








if __name__=="__main__":
    main()
