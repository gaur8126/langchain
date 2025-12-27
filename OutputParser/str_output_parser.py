from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="chat-comletion",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

llm = ChatHuggingFace(llm=llm1)

# prompt -> detailed report 
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result = llm.invoke(prompt1)
print(result.content)
print("-"*10)

prompt2 = template2.invoke({'text':result.content})

result1 = llm.invoke(prompt2)

print(result1.content)