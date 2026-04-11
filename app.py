# pyright: reportMissingImports=false

## RAG Q&A Conversation With PDF Including Chat History

import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- STREAMLIT UI ---------------- #

st.title("Conversational RAG With PDF uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Input Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

# ---------------- MAIN APP ---------------- #

if api_key:

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # Session ID to maintain chat memory
    session_id = st.text_input("Session ID", value="default_session")

    # Store chat history per session
    if "store" not in st.session_state:
        st.session_state.store = {}

    # Function to get chat history
    def get_session_history(session_id: str):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # Upload PDF files
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        documents = []

        # -------- LOAD PDFs -------- #
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"

            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # -------- SPLIT TEXT -------- #
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents)

        # -------- VECTOR STORE -------- #
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

        # -------- PROMPT FOR HISTORY-AWARE RETRIEVER -------- #
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, "
                       "rephrase it into a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # -------- HISTORY AWARE RETRIEVER -------- #
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        # -------- QA PROMPT -------- #
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based only on the provided context.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # -------- DOCUMENT CHAIN -------- #
        document_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # -------- FULL RAG CHAIN -------- #
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )

        # -------- ADD MEMORY (Important) -------- #
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # -------- CHAT INPUT -------- #
        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # -------- OUTPUT -------- #
            st.write("### Answer:")
            st.write(response["answer"])

else:
    st.warning("Please enter your Groq API key to use the application.")