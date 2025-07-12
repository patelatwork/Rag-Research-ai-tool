import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("Research Paper Q&A Tool")


uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

# ChatGPT-like UI
st.markdown("""
<style>
.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
    text-align: right;
    margin-left: 30%;
    margin-right: 0;
}
.bot-msg {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
    text-align: left;
    margin-right: 30%;
    margin-left: 0;
}
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstores' not in st.session_state:
    st.session_state.vectorstores = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if uploaded_file:
    if st.session_state.vectorstores is None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(docs)
        llmem = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstores = FAISS.from_documents(embedding=llmem, documents=chunks)
        retriever = vectorstores.as_retriever(search_kwargs={"k": 5})
        st.session_state.vectorstores = vectorstores
        st.session_state.retriever = retriever

st.markdown("---")

# Show chat history at the top
for chat in st.session_state.chat_history:
    st.markdown(f'<div class="user-msg">{chat["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">{chat["answer"]}</div>', unsafe_allow_html=True)

# Query box and Ask button below chat history
question = st.text_input("Ask a question about the research paper")
ask_button = st.button("Ask")

if ask_button and uploaded_file and question:
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )
    model = ChatOpenAI(model="gpt-4o",temperature=1.5)
    context = st.session_state.retriever.invoke(question)
    prompt2 = prompt.invoke({"context": context, "question": question})
    result = model.invoke(prompt2)
    st.session_state.chat_history.append({"question": question, "answer": result.content})
    st.rerun()