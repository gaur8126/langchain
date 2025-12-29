from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic'],
)


model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

parser = StrOutputParser()

chain = prompt | model | parser


result = chain.invoke({'topic':'cricket'})


print(result)

chain.get_graph().print_ascii()