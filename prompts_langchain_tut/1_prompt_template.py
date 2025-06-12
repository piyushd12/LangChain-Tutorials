from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=10
# )


llm = ChatGroq(model_name="SEE IN GROQ DEVELOPER CONSOLE") #Example: llama3-70b-8192



template = PromptTemplate(
    template="""
    what is the capital of {country}?
    """,
    input_variables=["country"],
    validate_template=True
)

# Cap = input("Enter the country: ")

text = template.invoke({"country": "india"})

res = llm.invoke(text)
print(res)
    
