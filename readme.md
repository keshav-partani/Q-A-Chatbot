
https://github.com/user-attachments/assets/93f14bac-0ba3-4711-8258-f5706ee27221
Uploading Chatbot.mp4…


Install all the dependencies by running
    pip install streamlit PyPDF2 langchain langchain-huggingface langchain-community langchain-groq python-dotenv

Paste your api key from
    Hugging_face_api_key = https://huggingface.co/settings/tokens
    Groq_api_key = https://console.groq.com/keys

If you need to change your document file, you can change by changing "PDF_PATH" variable in app.py

After all the setup you run app by running
    streamlit run app.py

(Wait till embedding is prepared then you can upload your queries)

