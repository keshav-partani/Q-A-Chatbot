import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Path to the PDF file
PDF_PATH = "Corpus.pdf"


# Function to process PDF and prepare the embeddings
@st.cache_resource
def prepare_document_search(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        raw_text = ''
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        model = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        )
        document_search = FAISS.from_texts(texts, embeddings)
        return document_search
    except Exception as e:
        st.error(f"Error preparing document search: {e}")
        return None


# Initialize the language model
@st.cache_resource
def initialize_llm():
    try:
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None


# Initialize the prompt template
@st.cache_resource
def initialize_prompt_template():
    return ChatPromptTemplate.from_messages([("human",
                                              "Answer directly as a human to the following question: {topic}. If the answer is not in the text, then say directly 'Please contact the business directly for this information.'")])


# Streamlit app
st.title('Ask a question')

# Prepare embeddings and initialize model and prompt
document_search = prepare_document_search(PDF_PATH)
llm = initialize_llm()
prompt_template = initialize_prompt_template()
if llm and prompt_template:
    chain = prompt_template | llm

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    new_prompt = prompt
    if len(st.session_state.messages) > 1:
        new_prompt += "\nPrevious Question: " + st.session_state.messages[-2]['content']

    # Perform similarity search
    if document_search:
        docs = document_search.similarity_search(new_prompt)
        search_result = "".join([doc.page_content for doc in docs])

        llm_input = search_result + "\n\nCurrent Question: " + new_prompt

        try:
            answer = chain.invoke(({"topic": llm_input})).content
            if "the text does not mention" in answer:
                response = "Please contact the business directly for this information."
            else:
                response = answer
        except Exception as e:
            response = f"Error in generating response: {e}"

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
    else:
        st.chat_message('assistant').markdown("Document search preparation failed.")