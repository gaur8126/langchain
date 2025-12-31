from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

"""Built in tools"""

# duckducktool
duck_tool = DuckDuckGoSearchRun()
duck_result = duck_tool.invoke('ipl news')

# print(duck_result)

# shelltool 
shell_tool = ShellTool()
# shell = shell_tool.invoke('whoami')
# print(shell)



"""Custom tools"""

from langchain_community.tools import tool

""" `Custom tooling is a three step process` """ 

"""Method 1. using @tool"""

# step-1 create a function 
def multiply(a,b):
    """Multiply two numbers"""
    return a*b

# step-2 add type hints 
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a*b


# step-3 add tool decorator 
@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a*b


# result = multiply.invoke({"a":3,"b":5})
# print(result)


"""Method 2. using Structuretool & Pydantic"""

from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultipyInput(BaseModel):
    a: int = Field(required = True, description="The first number to add")
    b: int = Field(required = True, description="The first number to add")

def multiply_func(a: int, b:int) -> int:
    return a*b


multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name='multiply',
    description="Multiply two numbers",
    args_schema=MultipyInput
)

# result = multiply_tool.invoke({'a':3,'b':3})
# print(result)

"""Method 3. using BaseTool class: is the abstract class for all tools in LangChain' """
from langchain_community.tools import BaseTool
from typing import Type

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultipyInput

    def _run(self, a: int, b: int) -> int:
        return a*b
    
multiply_tool = MultiplyTool()
result = multiply_tool.invoke({'a':3,'b':3})
print(result)