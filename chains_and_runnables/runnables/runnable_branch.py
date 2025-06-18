from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv()

model = ChatGroq(model_name='gemma2-9b-it')

report_gen_prompt = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

summarization_prompt = PromptTemplate(
    template='Summarize the following text:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = report_gen_prompt | model | parser
summary_chain = summarization_prompt | model | parser


branch_chain = RunnableBranch(
    (lambda report: len(report.split()) > 300, summary_chain),
    RunnableLambda(lambda x: "Summary not generated (report too short)")
)

final_chain = RunnableLambda(
    lambda x: {'topic': x['topic']}
) | report_gen_chain | RunnableLambda(
    lambda report: {
        'report': report,
        'word_count': len(report.split()),
        'summary': branch_chain.invoke(report)
    }
)

result = final_chain.invoke({'topic': 'Russia vs Ukraine'})

print("=== Report ===\n", result['report'], "\n")
print("=== Word Count ===\n", result['word_count'], "\n")
print("=== Summary ===\n", result['summary'])
