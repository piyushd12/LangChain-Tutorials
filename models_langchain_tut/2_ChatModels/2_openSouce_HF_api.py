from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

hf_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=10
)

chat_llm = ChatHuggingFace(llm=hf_llm)

response = chat_llm.invoke("What is the capital of India?")
print(response.content)
