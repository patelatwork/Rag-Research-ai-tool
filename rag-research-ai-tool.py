from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
load_dotenv()

#Specification of LLM
model = ChatOpenAI(model="gpt-4o")
llmem = OpenAIEmbeddings(model="text-embedding-3-large")
#Indexing Part
#1.Loading
loader = PyPDFLoader("D:\Programs\RAG application\Research Paper - Feature_Selection_Using_an_Improved_Gravitational_Search_Algorithm.pdf")
docs = loader.load()
#2.Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs)
#3.Create Embeddings and Store in vector stores(databases)
vectorstores = FAISS.from_documents(embedding=llmem,documents=chunks)
retriever = vectorstores.as_retriever(search_kwargs={"k": 5})

#Augmentation
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
context = retriever.invoke("What do we mean by linear kbest")

question = "What do we mean by linear kbest"
prompt2 = prompt.invoke({"context":context,"question":question})
result = model.invoke(prompt2)

print(result.content)