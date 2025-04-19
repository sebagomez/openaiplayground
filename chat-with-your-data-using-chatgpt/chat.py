import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

_ = load_dotenv(find_dotenv()) # read local .env file

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def print_output(docs):
    for doc in docs:
        if doc[0] == "content":
            print(doc[1]) 

def print_docs(docs):
    for doc in docs:
        print(f"Text: {doc.page_content}")
        print(f"Page: {doc.metadata['page']}")
        print("=========================================")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

# Load the document
loader = PyPDFLoader('data/Formula 1 - 2025 Season_ocr.pdf')
pages = loader.load()

# Split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)

# Create the embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Load it into the vector store and embed
vectordb = FAISS.from_documents(documents, embeddings_model)

vectordb.save_local("faiss2_index")

new_db = FAISS.load_local("faiss2_index", embeddings_model, allow_dangerous_deserialization=True)

retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

system_prompt = """Given the chat history and a recent user question \
generate a new standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed or otherwise return it as is."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever_with_history = create_history_aware_retriever(
    llm, retriever, prompt
)


qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)

chat_history = []

question = input("Please enter your question: (or type 'exit' to finish the conversation) ")
while question.lower() != "exit":
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})

    chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])

    print(ai_msg["answer"])
    question = input("Please enter your question: (or type 'exit' to finish the conversation) ")
