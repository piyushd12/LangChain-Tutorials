from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()

model = ChatGroq(model_name="gemma2-9b-it")

# This enforces the output to follow a specific schema but does not provide output validation.

# schema = [
#     ResponseSchema(name='title', description='Title of the report'),
#     ResponseSchema(name='summary', description='A brief summary of the report'),
#     ResponseSchema(name='content', description='Detailed content of the report'),
#     ResponseSchema(name='conclusion', description='Conclusion of the report')
# ]

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# template1 = PromptTemplate(
#     template='Write a detailed report on {topic} \n {format_instructions}',
#     input_variables=['topic'],
#     partial_variables={'format_instructions': parser.get_format_instructions()}
# )

chain = template1 | model | parser

result = chain.invoke({'topic': "Artificial Intelligence"})

# print(result)

print(json.dumps(result, indent=2))