from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from typing import List,Any
import asyncio
import os
from logger import ( log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()
tavily_tool = TavilyMap(max_depth=3, max_breadth=50, max_pages=1000)
tavily_extract = TavilyExtract()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024,chunk_size=50)
vecotor_store = PineconeVectorStore(embedding=embedding,index_name=os.environ.get("PINECONE_INDEX_NAME"))

async def index_document_async(all_docs:List[Document],batch_size:int=20):
    log_info("indexing started...")
    batches = [
        all_docs[i : i + batch_size] for i in range(0, len(all_docs), batch_size)
    ]
    log_info(f"batches are created {len(batches)} from {len(all_docs) } documents.")
    async def add_batch(batch:List[Document]):
        try:
            vecotor_store.add_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch - {e}")
            return False
        return True

    tasks = [add_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks,return_exceptions=True)
    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )

def document_splitter(docs:List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    splitted_docs = text_splitter.split_documents(docs)
    return splitted_docs

def get_urls(url:str):
    sitemap = tavily_tool.invoke({"url":"https://docs.langchain.com/"})
    urls = sitemap.get("results")
    return urls

async def async_extraction(urls: List[str], chunk_size: int = 10) -> List[Document]:
    url_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    tasks = [docs_extraction(chunk) for chunk in url_chunks]
    results = await asyncio.gather(*tasks)
    return [doc for sublist in results for doc in sublist] 


async def docs_extraction(urls:List[str]):
    log_info(f"Docs extraction started")
    all_docs = []
    try:
        res = await asyncio.to_thread(tavily_extract.invoke,{"urls":urls,"max_depth":3,"extract_depth":"advanced"})
        for item in res['results']:
            if(item.get("raw_content")):
                doc = Document(page_content=item['raw_content'],metadata={"source":item['url']})
                all_docs.append(doc)
        
    except Exception as e:
        log_error(f"exception in docs_extraction: {e}")
        

    return all_docs
    

async def main():
    urls = get_urls("https://docs.langchain.com/")
    log_info(f"sitemap completed:{len(urls)}")
    docs =await async_extraction(urls=urls)
    log_info(f"asynch docs_extraction completed: {len(docs)}")
    splitted_docs = document_splitter(docs=docs)
    log_success( f"Text Splitter: Created {len(splitted_docs)} chunks from {len(docs)} documents")
    await index_document_async(splitted_docs)

if __name__=="__main__":
    asyncio.run(main())
    