from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

os.environ['HF_HOME'] = '/home/piyush/Documents/model_cache/hf_cache'

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
               model_name = "sentence-transformers/all-mpnet-base-v2"      
)

text = "This is a test sentence for embedding."

vector = embedding_model.embed_query(text)

document = [
               "This is a test sentence for embedding.",
               "This is another sentence for embedding.",
               "This is yet another sentence for embedding."
]

vector = embedding_model.embed_documents(document)

print(vector)