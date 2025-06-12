from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
import os

os.environ['HF_HOME'] = 'Path to the cache directory.' #Example: Documents/model_cache/hf_cache

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='Model ID', #Example: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    task='text-generation',
    pipeline_kwargs=dict(
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")
    print(f"You: {user_input}")
    chat_history.append(user_input)
    if user_input.lower() == "exit" : break
    res = model.invoke(chat_history)
    chat_history.append(res)
    print(f"AI: {res}")

print(chat_history)