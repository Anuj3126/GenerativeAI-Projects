from dotenv import load_dotenv
load_dotenv() #Loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

##Function to load Gemini Pro model and get responses

model=genai.GenerativeModel("gemini-pro")
def get_gemini_response(query):
    response=model.generate_content(query)
    return response.text

### INITIALIZING STREAMLIT APP
st.set_page_config(page_title="Gemini Q&A")
st.header("Gemini Chatbot")
input=st.text_input("Input: ",key="input")
submit=st.button("Ask Question :)")

#When submit is clicked

if submit:
    response=get_gemini_response(input)
    st.subheader("Response: ")
    st.write(response)