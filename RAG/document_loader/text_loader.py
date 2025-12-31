from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os

loader = TextLoader('cricket.txt',encoding='utf-8')

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = os.getenv("GEMINI_API_KEY")
    
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "write a summary for the following poem - \n {poem}.",
        ),
        # ("human", "{input}"),
        

    ]
)

parser = StrOutputParser()

docs  = loader.load()

print(docs[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))