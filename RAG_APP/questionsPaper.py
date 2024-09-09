import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# Define the Streamlit app


def main():
    st.title("Advanced RAG PDF Document Search with Conditional Chains")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

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

            vectorstore = FAISS.from_documents(
                documents, huggingface_embeddings)

            st.success("PDF Loaded and Processed Successfully!")

            # Initialize the LLM for various tasks
            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=150,
                api_key=os.getenv("GROQ_API_KEY")
            )

            # Define separate prompts
            summary_prompt_template = """
            You are an assistant that provides a summary of the document content. 
            Given the following content, provide a concise summary.

            Document Content:
            {content}

            Summary:
            """

            questions_prompt_template = """
            You are an assistant that generates insightful questions from the text. 
            Given the following content from a document, create a list of relevant questions.

            Document Content:
            {content}

            Questions:
            """

            q_and_a_prompt_template = """
            You are an assistant that answers questions based on the text. 
            Given the following content from a document and a question, provide a precise answer.

            Document Content:
            {content}

            Question:
            {question}

            Answer:
            """

            # Create chains for each task
            summary_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["content"],
                    template=summary_prompt_template
                )
            )

            questions_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["content"],
                    template=questions_prompt_template
                )
            )

            q_and_a_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["content", "question"],
                    template=q_and_a_prompt_template
                )
            )

            # Allow user to enter a query
            query = st.text_input("Enter your query")
            button = st.button("Submit Query")

            if button:
                if query:
                    # Perform a retrieval search to find relevant chunks
                    relevant_documents = vectorstore.similarity_search(query)
                    if relevant_documents:
                        # Combine relevant content from the chunks
                        combined_content = "\n".join(
                            [doc.page_content for doc in relevant_documents])

                        if "summarize" in query.lower():
                            # Generate Summary
                            summary_result = summary_chain.run(
                                content=combined_content
                            )
                            st.write("Summary:")
                            st.write(summary_result)

                        elif "question" in query.lower():
                            # Generate Questions
                            questions_result = questions_chain.run(
                                content=combined_content
                            )
                            questions = questions_result.split("\n")
                            q_and_a_results = []

                            for question in questions:
                                if question.strip():
                                    answer = q_and_a_chain.run(
                                        content=combined_content,
                                        question=question.strip()
                                    )
                                    q_and_a_results.append((question, answer))

                            st.write("Generated Questions and Answers:")
                            for q, a in q_and_a_results:
                                st.write(f"Question: {q}")
                                st.write(f"Answer: {a}")
                                st.write("-----")

                        elif "answer" in query.lower():
                            # Use a specific question from the user query
                            specific_question = query.replace(
                                "answer", "").strip()
                            if specific_question:
                                answer_result = q_and_a_chain.run(
                                    content=combined_content,
                                    question=specific_question
                                )
                                st.write("Answer:")
                                st.write(answer_result)
                            else:
                                st.write(
                                    "Please specify a question for an answer.")

                        else:
                            st.write(
                                "Invalid query. Please specify 'summarize', 'question', or 'answer'.")

                    else:
                        st.write("No relevant document found.")

            # Clean up the temporary file
            os.remove("temp.pdf")


if __name__ == "__main__":
    main()
