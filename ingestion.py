from dotenv import load_dotenv
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from typing import List,Any
import asyncio

tavily_tool = TavilyMap(max_depth=1, max_breadth=20, max_pages=1000)
tavily_extract = TavilyExtract()

async def main():
    print("Ingestion for Documentation Helper!")
    tavily_tool_result = tavily_tool.invoke({"url":"https://beingguru.com/"})
    results = tavily_tool_result.get("results")
    print(f"tavily_tool_result: {results}")
    print(f"fetch results: {len(results)}")
    url_batches = chunk_urls(results)
    print(f"URL chunks created: {len(url_batches)}")
    all_pages = await async_extraction(url_batches)
    print(f"async_extraction: {len(all_pages)}")

    


async def async_extract(urls:List[str])->List[dict[str,Any]]:
    response = await asyncio.to_thread(tavily_extract.invoke,{"urls":urls})
    return response['results']
    
async def async_extraction(url_batches:List[List[str]]):
    print("aync_extraction...")
    tasks = [async_extract(batch) for batch in url_batches]
    results = await asyncio.gather(*tasks)
  
    all_docs = []
    for result in results:
        for page in result:
            document = Document(page_content=page['raw_content'],metadata={"source":page['url']})
            all_docs.append(document)

    return all_docs



def chunk_urls(urls:List[str], chunk_size:int=10 )->List[str]:
    print("chunking urls...")
    chunks = []
    for i in range(0,len(urls),chunk_size):
        chunk = urls[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

if __name__=="__main__":
    asyncio.run(main())
    