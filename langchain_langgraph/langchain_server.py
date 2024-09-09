import uvicorn
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-pro',
                               api_key=os.getenv('GEMINI_API_KEY'),
                               convert_system_message_to_human=True)
# useful to fetch only string from response
parser = StrOutputParser()
# parser.invoke(response)
# step01
prompt_template = ChatPromptTemplate(
    [
        ("system", "Translate the following into {language}"),
        ("user", "{text}")

    ]
)
chain = prompt_template | model | parser

app = FastAPI(
    title='My LLm APP',
    description="My LLm APP",
    version="1.0"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
