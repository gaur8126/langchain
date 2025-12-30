from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
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

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2, model, parser)
})


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))