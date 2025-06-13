from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import json

load_dotenv()

model = ChatGroq(model_name="gemma2-9b-it")

# This enforces the output to follow a specific schema and provides output validation.

class Person(BaseModel):

    name : str = Field(description="Name of the person")
    age : int = Field(description="Age of the person", gt=21, lt=70)
    city : str = Field(description="City where the person lives")
    occupation : str = Field(description="Occupation of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template1 = PromptTemplate(
    template='Give me details of a person living in {country} \n {format_instruction}',
    input_variables=['country'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'country': 'canada'})

print(result)
