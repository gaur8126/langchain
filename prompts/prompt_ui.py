from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name",["Attention is All You Need",
                                                         "BERT: Pre-train of Deep Bidirectional Transformers",
                                                         "GPT-3:Language Models are Few Short Learners"])

style_input = st.selectbox("Select Explaination Style", ["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explaination length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (Detailed explaination)"])

#template

template = PromptTemplate(
    template="""
Please summerize the research paper titled "{paper_input}" with following specifications:
Explaination Style: {style_input}
Explaination Length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical eqations if present in the paper.
    - Explain the mathematical concept using simple, intuitive code snippets where
    applicable.
2. Analogies:
    - Use relatable analogy to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available
in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables= ['paper_input','style_input','length_input']
)

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input' : style_input,
    'length_input' : length_input
})


if st.button('Summerize'):
    result = model.invoke(prompt)
    st.write(result.content)