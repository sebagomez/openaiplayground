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

question = "Who won the first GP of the 2025 Formula 1 season?"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

docs = llm.invoke(question) 

print_output(docs)

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

# print(vectordb.index.ntotal)

vectordb.save_local("faiss2_index")

new_db = FAISS.load_local("faiss2_index", embeddings_model, allow_dangerous_deserialization=True)

# docs = new_db.similarity_search(question)

# print_docs(docs)

retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

prompt = hub.pull("rlm/rag-prompt")


#combine multiple steps in a single chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() #convert the chat message to a string
)

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)