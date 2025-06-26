from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

doc_loader = DirectoryLoader(
    path='documentLoaders/PDFs',
    glob='*.pdf',
    # show_progress=True,
    # loader_cls=PyPDFLoader
)

docs = doc_loader.lazy_load()
for doc in docs:
    print(doc.metadata)
