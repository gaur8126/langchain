from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="chat-comletion",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm1)

schema = [
    ResponseSchema(name='fact1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact3',description='Fact 3 about the topic')
]


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)