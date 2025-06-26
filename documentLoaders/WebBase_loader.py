from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

prompt = PromptTemplate(
    template="Summarize the page content in 500 words \n {page_content}",
    input_variables=["page_content"]
)

parser = StrOutputParser()

urls = ["https://www.tle-eliminators.com/cp-sheet","https://www.dataversity.net/a-brief-history-of-machine-learning/"]

loader = WebBaseLoader(web_path=urls)

doc = loader.load()

# print(len(doc))

chain = prompt | model | parser

res = chain.invoke({'page_content': doc[1]})

print(res)