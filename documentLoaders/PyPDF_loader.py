from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

doc_loader = PyPDFLoader(file_path="documentLoaders/PDFs/numbers_1_to_100.pdf")

doc_load = doc_loader.load()

# print(type(doc_load))
# print(len(doc_load))

# print(doc_load[0].metadata)


pages = []
# pages.append(doc_load)

doc_lazy_load = doc_loader.lazy_load()
# print(type(doc_lazy_load))

# print(doc_lazy_load)

for doc in doc_lazy_load:
    pages.append(doc)

prompt = PromptTemplate(
    template="Tell me on which page does this number {number} appears from following document - \n {pdf_content}",
    input_variables=["number","pdf_content"]
)

chain = prompt | model | StrOutputParser()

res = chain.invoke({"number":53,"pdf_content": pages})

print(res)  

# for page in pages:
#     print(page)