from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Model ID", #Example: HuggingFaceH4/zephyr-7b-beta
    task="text-generation",
    max_new_tokens=50,
    temperature=1.5
)

prompt = load_prompt('template.json')

chain = prompt | llm


paper_input = input("Enter the paper: ")
style_input = input("Enter the style: ")
length_input = input("Enter the length: ")

res = chain.invoke({
    "paper_input" : paper_input,
    "style_input" : style_input,
    "length_input" : length_input
})

print(res)

