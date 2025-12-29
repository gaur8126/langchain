from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the text \n {text}',
    input_variables=['text']
)


model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser


result = chain.invoke({'topic':'cricket'})


print(result)

chain.get_graph().print_ascii()