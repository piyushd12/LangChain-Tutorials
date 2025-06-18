from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

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

# RunnableSequence is same as the '|' (pipe) operator. 
# Example: RunnableSequence(prompt,model,parser) = prompt | model | parser  (like sequential chains)

chain = RunnableSequence(template1,model,parser,template2,model,parser)
# chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Artificial Intelligence"})
print(result)   

