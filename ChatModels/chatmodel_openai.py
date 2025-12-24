import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

response = llm.invoke("who was the main character in the game of thrones series ?")

print(response.content)