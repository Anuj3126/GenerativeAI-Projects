from dotenv import load_dotenv
load_dotenv() #Loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Gemini Pro Vision model and get responses related to images
model=genai.GenerativeModel("gemini-pro-vision")
def get_gemini_response(image,query):
    if input!="":
        response=model.generate_content([image,query])
    response=model.generate_content(image)
    return response.text

#INITIALIZING STREAMLIT APP

st.set_page_config(page_title="Gemini Vision Pro")
st.header("Gemini Chatbot with images")
input=st.text_input("Input Prompt: ",key="input")

uploaded_file=st.file_uploader("Choose an image: ",type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image.",use_column_width=True)

submit=st.button("Search :)")

# If submit it clicked
if submit:# Because it will only work if an image is submitted
    response=get_gemini_response(image,input)
    st.subheader("Response: ")
    st.write(response)