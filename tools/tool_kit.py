from langchain_core.tools import tool

# custom tools 
@tool
def add(a:int, b:int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b

class MathToolKit:
    def get_tools(self):
        return [add, multiply]