from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from logger import log_error,log_info,log_success
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import Dict,Any
from langchain.agents import create_agent
from langchain.messages import ToolMessage
import asyncio
import os


load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
vecotor_store = PineconeVectorStore(embedding=embedding,index_name=os.environ.get("PINECONE_INDEX_NAME"))

model = init_chat_model(model="gpt-5.2",model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query:str):

    """Retrieve relevant documentation to help answer user queries about LangChain."""

    retrieved_docs = vecotor_store.as_retriever().invoke(query,k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    log_info("context retreived from vector db.")
    return retrieved_docs, serialized


def run_llm(query:str)->Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The users's question
    
    Return:
        Dictionary containing:
            - answer: The generated answer
            - context: The list of retrieved documents
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model=model,tools=[retrieve_context],system_prompt=system_prompt)

    messages = [{"role":"user","content":query}]

    log_info("Agent is invoked.")
    response = agent.invoke({"messages":messages})

    answer = response['messages'][-1].content

    log_success(f"answer retrieved from agent: {answer}")

    context_docs = []
    for message in response['messages']:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.append(message.artifact)

    return {"answer":answer,"context":context_docs  }




def main():
    log_info("RAG retriever is in-action...")
    result = run_llm(query="What are agents?")
    log_info("llm results received.")
    print(result)


if __name__ == "__main__":
    #asyncio.run(main())
    main()