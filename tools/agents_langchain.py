import os 
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from dotenv import load_dotenv
load_dotenv()

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('top news in india today')

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

# STEP 2: Pull the react prompt from Langchain Hub
prompt = hub.pull('hwchase17/react') # pulls the standard React agent prompt

# step 3: create the react agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

# step 5: Invoke

response = agent_executor.invoke({"input":"3 ways to reach goa from delhi"})

print(response)