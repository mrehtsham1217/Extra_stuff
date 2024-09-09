import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

# Define the Streamlit app


def main():
    # Custom CSS styling
    st.markdown(
        """
        <style>
        body {
            background-color: #000435;
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 3em;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 1em;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            margin-bottom: 1em;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .result-box {
            border: 1px solid #dddddd;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 10px;
            margin-top: 1em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page title with custom styling
    st.markdown('<h1 class="title">RAG Application powered by Groq & Vector Dbs</h1>',
                unsafe_allow_html=True)

    # Upload PDF section with custom styling
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("Loading and processing the PDF..."):
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the PDF document
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                separators=['\n', '\n\n', ','],
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(documents)

            # Display the number of chunks
            st.write(f"Number of chunks created: {len(documents)}")

            # Embed the document chunks
            huggingface_embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            embeddings = [huggingface_embeddings.embed_query(
                doc.page_content) for doc in documents]

            # Build the Chroma vectorstore
            vectorstore = Chroma.from_documents(
                documents, huggingface_embeddings)

            st.success("PDF Loaded and Processed Successfully!")

            # Initialize the LLM
            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=150,
                api_key=os.getenv("GROQ_API_KEY")
            )

            # Create the RetrievalQA chain
            retriever = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            # Allow user to enter a query with a stylish input box
            query = st.text_input("Enter your query")
            button = st.markdown(
                '<button class="btn">Find Answer</button>', unsafe_allow_html=True)

            if button:
                if query:
                    # Perform retrieval and LLM-based QA
                    result = retriever.invoke(query)

                    # Display the generated answer in a styled box
                    st.markdown('<div class="result-box">',
                                unsafe_allow_html=True)
                    st.write("Generated Answer:")
                    st.write(result)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Clean up the temporary file
            os.remove("temp.pdf")


if __name__ == "__main__":
    main()
