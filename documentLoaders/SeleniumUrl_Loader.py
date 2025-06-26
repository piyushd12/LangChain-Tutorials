from langchain_groq import ChatGroq
from langchain_community.document_loaders import SeleniumURLLoader
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

urls = ["https://www.tle-eliminators.com/cp-sheet","https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.url_selenium.SeleniumURLLoader.html"]

loader = SeleniumURLLoader(urls=urls)

doc = loader.load()
print(doc[0])