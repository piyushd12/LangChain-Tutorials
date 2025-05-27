from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat = ChatGroq(model_name="llama3-70b-8192")
res = chat.invoke("What is the capital of France?")

print(res.content)
