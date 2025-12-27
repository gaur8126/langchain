from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
import os
load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="chat-comletion",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

llm = ChatHuggingFace(llm=llm1)

parser = JsonOutputParser()

# prompt -> detailed report 
template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()
# result = llm.invoke(prompt)


# final_result = parser.parse(result.content)
# print(final_result)

chain = template | llm | parser

result = chain.invoke({})

print(result)