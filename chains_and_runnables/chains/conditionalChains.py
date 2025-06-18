from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from typing import Literal

load_dotenv()

model = ChatGroq(model_name="gemma2-9b-it")

class outputFormatter(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of the text")

parser = PydanticOutputParser(pydantic_object=outputFormatter)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative:\n\n{feedback}\n\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser

prompt2 = PromptTemplate(
    template='Write an appropriate response to this {sentiment} feedback:\n\n{feedback}',
    input_variables=['feedback', 'sentiment']
)

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] in ['positive', 'negative'], prompt2 | model | StrOutputParser()),
    RunnableLambda(lambda x: f"No response needed for sentiment: {x['sentiment']}")
)

full_chain = RunnableLambda(lambda x: {
    "feedback": x["feedback"],
    "sentiment": classifier_chain.invoke({"feedback": x["feedback"]}).sentiment
}) | branch_chain

feedback_text = "ughh! the battery life is terrible, I am not happy with this product"
result = full_chain.invoke({"feedback": feedback_text})
print(result)   