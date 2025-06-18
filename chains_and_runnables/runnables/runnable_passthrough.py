from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import json

load_dotenv()

model = ChatGroq(model_name='gemma2-9b-it')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# RunnablePassthrough is used to pass the output of one chain to the input of another chain
# Example: RunnablePassthrough()
# the output of the chain will be passed to the input of the next chain 
# It does not do any processing of the input. It just passes the input to the next chain.

joke_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : prompt2 | model | parser
})

# It passed the joke which is the output of the first chain (i.e. joke_gen_chain) to the second chain (i.e. final_chain)
# So that's why we can print or use the joke in the second chain

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic' : 'Earth'})

print(json.dumps(result, indent=2))

final_chain.get_graph().print_ascii()


