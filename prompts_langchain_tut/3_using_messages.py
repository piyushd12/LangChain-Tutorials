from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

chat_history = [
    SystemMessage(content="Your name is XYZ and you are a helpful AI chat assistant.")
]

while True:
    user_input = input("You: ")
    print(f"You: {user_input}")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit": break
    res = model.invoke(chat_history)
    chat_history.append(AIMessage(content=res.content))
    print(f"AI: {res.content}")

print(chat_history)
