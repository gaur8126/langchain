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

response = llm.invoke("")

print(response.content)


# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# llm = ChatOpenAI(
#     model="openai/gtp-4o",  # replace with 4.5 when available
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0.7,
#     max_tokens=512
# )

# response = llm.invoke("which company own you ?")
# print(response.content)
