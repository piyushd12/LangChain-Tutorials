from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

# chat_template = ChatPromptTemplate([
#     ('system' , "Your name is ABCDE and you are a helpful AI chat assistant."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ('human' , "{user_input}")
# ])

chat_template = ChatPromptTemplate([
    ('system' , "Your System Prompt"), #Example: You are a helpful AI chat assistant.
    MessagesPlaceholder(variable_name="chat_history"),
    ('human' , "{user_input}")
])

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit' : 
        print(prompt,'\n\n\n')
        break
    prompt = chat_template.invoke({
        'chat_history' : chat_history,
        'user_input' : user_input
    })
    res = model.invoke(prompt)
    chat_history.append(AIMessage(content=res.content))
    print(f"AI: {res.content}")

print(chat_template)
