import random

class NakliLLM:

    def __init__(self):
        print("LLM created")

    def predict(self, prompt):

        response_list = [
            'Delhi is the capital of india',
            'IPL is a cricket league',
            'AI stands for Artifitial Intelligence'
        ]

        return {'response': random.choice(response_list)}
    

llm = NakliLLM()

# print(llm.predict("What is the capital of india"))


class NakliPromptTemplate:

    def __init__(self,template,input_variable):
        self.template = template
        self.input_variable = input_variable

    def format_(self, input_dict):
        return self.template.format(**input_dict)
    
template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variable=['length','topic']
)

prompt = template.format_({'length':'short','topic':'india'})
        

class NakliChain:
    def __init__(self,llm,prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):

        final_prompt = self.prompt.format_(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']
    
chain = NakliChain(llm, template)

print(chain.run({'length':'short','topic':'india'}))