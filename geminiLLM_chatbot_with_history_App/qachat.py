from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai

def load_api_key():
    """Loads the Google API key from the .env file."""
    try:
        load_dotenv()
        return os.getenv("GOOGLE_API_KEY")
    except FileNotFoundError:
        st.error("Error: .env file not found. Please create a .env file with your Google API key.")
    return None


## Function to load Gemini Pro model and get response
def get_gemini_response(query, model):
    """Sends the query to the Gemini Pro model and returns the response.

    Args:
        query: The user's query string.
        model: The loaded GenerativeModel object.

    Returns:
        The response object from the Gemini Pro model or None if there's an error.
    """
    try:
        response = chat.send_message(query, stream=True)
        return response
    except Exception as e:
        st.error(f"Error: An error occurred while processing the query: {e}")
    return None

## Streamlit App Initialization
st.set_page_config(page_title="Chatbot")
st.header("Chatbot Q&A")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Load API key (handle potential errors)
api_key = load_api_key()
if not api_key:
    st.stop()  # Stop app execution if API key is not loaded

# Configure genai if API key is loaded
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    chat=model.start_chat(history=[])

## User Input and Submit Button
input_field = st.text_input("Input:", key="input")
submit_button = st.button("Ask Question :)")

if submit_button and input_field:
    if not model:  # Check if model is loaded (API key issue)
        st.error("Error: Could not load the Gemini Pro model. Please check your API key.")
        st.stop()

    # Get response and update history
    with st.spinner("Generating response..."):#Loading instructor..shows the response is being loading
        response = get_gemini_response(input_field, model)
    if response:
        st.session_state['chat_history'].append(("You", input_field))
        st.subheader("Response: ")
        for chunk in response:
            try:
                st.write(chunk.text)
            except Exception as e:
                st.error(f"We encountered an issue processing your request. Please try rephrasing your question or try again later.")

            st.session_state['chat_history'].append(("Bot", chunk.text))
    else:
        st.warning("An error occurred while processing your request. Please try again.")

## Chat History Display
st.subheader("Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")



