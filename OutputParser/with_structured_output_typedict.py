from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from typing import TypedDict


model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

#schema 

class Review(TypedDict):
    summary: str
    sentimant: str

structured_model = model.with_structured_output(Review)

results = structured_model.invoke("""Perfect for combination to oily skin that is prone to congestion or acne. 
Cleanses gently, without stripping the skin off moisture. Washes off is seconds and doesn’t leave any residue.
A little goes a long way to build foam. Has a mild almost absent scent which is good for sensitive noses. It’s expensive but worthy. 
Definitely repurchasing.""")

print(results)
print(results['summary'])
print(results['sentimant'])
