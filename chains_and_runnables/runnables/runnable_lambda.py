from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
import json

load_dotenv()

model = ChatGroq(model='gemma2-9b-it')

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

# RunnableLambda is used to create a custom python function that can be used in the chain as runnables
# Example: RunnableLambda(lambda_function)

def word_count(text): 
    return {
        'joke' : text,
        'word_count' : len(text.split())
    }

word_count = RunnableLambda(word_count)

joke_gen_chain = prompt | model | parser

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(word_count)
# })

final_chain = joke_gen_chain | word_count

result = final_chain.invoke({'topic':'Earth'})

print(json.dumps(result,indent=2))
