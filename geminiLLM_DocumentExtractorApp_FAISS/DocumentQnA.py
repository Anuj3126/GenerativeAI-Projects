import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings #For embeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant specialized in extracting and summarizing information from documents.
    Please provide a detailed and well-explained answer to the user's question based on the given context.

    Context:
    {context}

    Question:
    {question}

    Guidelines:
    1. Use the context provided to construct your answer.
    2. If the context doesn't contain all the information, infer logical conclusions where possible.
    3. If the context is insufficient, indicate what information might be missing or needed.
    4. Structure your response in clear, concise, and coherent sentences.
    5. Highlight key points and important details.
    6. Provide examples from the context if relevant.

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response=chain.invoke(
        {
            "input_documents":docs,
            "question":user_question
        },
        return_only_outputs=True
    )

    print(response)
    st.write("Response: ",response["output_text"])

def main():
    st.set_page_config("VIT Chatbot")
    st.header("Gemini VIT chatbot :))")

    user_question= st.text_input("Ask a question from the pdf:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF files and click on the Submit & Process button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__=="__main__":
    main()