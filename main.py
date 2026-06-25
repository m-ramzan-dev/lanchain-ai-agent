from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)

def main():
    print("Hello from ecommerce-agent!")
    index_name = os.environ.get('PINECONE_INDEX_NAME')
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    question = 'If client does not assign work to ramzan, then what will be option to exit?'
    #retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    #docs = retriever.invoke(question)

    docs = vector_store.similarity_search(query=question)

    print(f"search_results: {docs}")
    context = format_doc(docs)
    messages = prompt.format_messages(context=context,question=question)
    llm_response = llm.invoke(messages)

    print(f"llm_response: {llm_response.content}")



def format_doc(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
    main()
