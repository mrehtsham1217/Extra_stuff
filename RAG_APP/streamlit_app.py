import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Define the Streamlit app
# BAse on cosine similarity seaching happens on PDF.
# Semantuc search-->RAG applications works on these features.


def main():
    """#+
    This function is the main entry point for the PDF document search application using LangChain.#+
    It handles the user interface, PDF loading, document processing, embedding, vector store creation,#+
    and query execution.
    Parameters:#+
    None
    Returns:#+
    None#+
    """
    st.title("PDF Document Search with LangChain")  # +
# +
    # Upload PDF#+
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")  # +
# +
    if uploaded_file is not None:  # +
        with st.spinner("Loading and processing the PDF..."):  # +
            # Save the uploaded file temporarily#+
            with open("temp.pdf", "wb") as f:  # +
                f.write(uploaded_file.getbuffer())  # +
                # convert the file into buffers#+
# +
            # Load the PDF document#+
            loader = PyPDFLoader("temp.pdf")  # +
            documents = loader.load()  # +
# +
            # Split the document into chunks#+
            text_splitter = RecursiveCharacterTextSplitter(  # +
                chunk_size=1000,  # +
                separators=['\n', '\n\n', ','],  # +
                chunk_overlap=200  # +
            )  # +
            documents = text_splitter.split_documents(documents)  # +
# +
            # Display the number of chunks#+
            st.write(f"Number of chunks created: {len(documents)}")  # +
# +
            # Embed the document chunks#+
            huggingface_embeddings = HuggingFaceEmbeddings(  # +
                model_name="all-MiniLM-L6-v2",  # +
                # if you are using cpu#+
                model_kwargs={'device': 'cpu'},  # +
                encode_kwargs={'normalize_embeddings': True}  # +
            )  # +
# +
            embeddings = [huggingface_embeddings.embed_query(  # +
                doc.page_content) for doc in documents]  # +
# +
            # Build the FAISS vectorstore#+
            vectorstore = FAISS.from_documents(  # +
                documents, huggingface_embeddings)  # +
# +
            st.success("PDF Loaded and Processed Successfully!")  # +
# +
            # Allow user to enter a query#+
            query = st.text_input("Enter your query")  # +
            button = st.button("Find Answer")  # +
# +
            if button:  # +
                if query:  # +
                    # Perform similarity search#+
                    relevant_documents = vectorstore.similarity_search(
                        query)  # +
                    if relevant_documents:  # +
                        st.write("Relevant Document Chunk:")  # +
                        st.write(relevant_documents[0].page_content)  # +
                    else:  # +
                        st.write("No relevant document found.")  # +
# +
            # Clean up the temporary file#+
            os.remove("temp.pdf")  # +
# >>>>>>> Tabnine >>>>>>>


def main():
    st.title("PDF Document Search with LangChain")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Loading and processing the PDF..."):
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
                # convert the file into buffers

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
                # if you are using cpu
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            embeddings = [huggingface_embeddings.embed_query(
                doc.page_content) for doc in documents]

            # Build the FAISS vectorstore
            vectorstore = FAISS.from_documents(
                documents, huggingface_embeddings)

            st.success("PDF Loaded and Processed Successfully!")

            # Allow user to enter a query
            query = st.text_input("Enter your query")
            button = st.button("Find Answer")

            if button:
                if query:
                    # Perform similarity search
                    relevant_documents = vectorstore.similarity_search(query)
                    if relevant_documents:
                        st.write("Relevant Document Chunk:")
                        st.write(relevant_documents[0].page_content)
                    else:
                        st.write("No relevant document found.")

            # Clean up the temporary file
            os.remove("temp.pdf")


if __name__ == "__main__":
    main()
