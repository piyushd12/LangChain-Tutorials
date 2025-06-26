from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="gemma2-9b-it")

doc_loader = TextLoader(file_path="documentLoaders/text_files/cricket.txt",encoding="utf-8")

doc = doc_loader.load()

# print(type(doc))
# print(len(doc))
# print(doc[0])

class outputFormatter(BaseModel):
    sports : str = Field(description="The sports mentioned in the poem")

parser = PydanticOutputParser(pydantic_object=outputFormatter)

prompt = PromptTemplate(
    template='Tell about which sports this poem is about - \n {poem} and follow these instructions {output_instructions}',
    input_variables=['poem'],
    partial_variables={'output_instructions': parser.get_format_instructions()}
)

chain = prompt | model | parser

res = chain.invoke({'poem' : doc[0].page_content})
print(res.sports)


