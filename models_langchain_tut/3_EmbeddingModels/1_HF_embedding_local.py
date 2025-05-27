# This is for models that are runnnig on local machine.
# This will download the model on local machine and then use it.

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

os.environ['HF_HOME'] = 'PATH TO THE CACHE DIRECTORY' #Example: Documents/model_cache/hf_cache

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
               model_name = "sentence-transformers/all-mpnet-base-v2" #Change model according to your need   
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