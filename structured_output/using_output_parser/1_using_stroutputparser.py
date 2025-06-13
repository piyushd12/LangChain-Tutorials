from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model_name="gemma2-9b-it")

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = "write 5 line summary of following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# Without using chains and parser

# prompt1 = template1.invoke({'topic':"Artificial Intelligence"})
# res1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text' : res1.content})
# res2 = model.invoke(prompt2)
# print(res2.content)

# using parser and chains

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Artificial Intelligence"})
print(result)   

