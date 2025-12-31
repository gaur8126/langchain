from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

# tool create 

@tool
def mulyiply(a:int, b:int) -> int:
    """Given 2 numbers a and b this tool returns their product."""
    return a * b

#tool binding 

llm_with_tools = llm.bind_tools([mulyiply])

print(llm_with_tools.invoke("can you multiply 3 with 10 ?"))