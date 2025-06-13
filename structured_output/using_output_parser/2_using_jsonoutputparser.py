from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()

model = ChatGroq(model_name="gemma2-9b-it")

# This gives output in json format but does not enforce schema

parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'topic': "Artificial Intelligence"})

print(json.dumps(result, indent=2))