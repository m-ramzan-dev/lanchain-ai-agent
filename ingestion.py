from dotenv import load_dotenv
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap



def main():
    print("Ingestion for Documentation Helper!")
    tavily_tool = TavilyCrawl()
    tavily_tool_result = tavily_tool.invoke({"url":"https://khanquranacademy.com/"})
    print(f"tavily_tool_result: {tavily_tool_result.get("results")}")
    



if __name__=="__main__":
    main()