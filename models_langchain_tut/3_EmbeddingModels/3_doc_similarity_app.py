from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ['HF_HOME'] = 'PATH TO THE CACHE DIRECTORY' #Example: Documents/model_cache/hf_cache


model = HuggingFaceEmbeddings(
               model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'who is jassi'

doc_embeddings = model.embed_documents(documents)
query_embedding = model.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)

# max_score = 0
# idx = 0
# for i in range(len(documents)):
#                if scores[0][i] > max_score:
#                        max_score = scores[0][i]
#                        idx = i
# print(scores)
# print(idx)

index, score = max(enumerate(scores[0]), key=lambda x: x[1])

print(f"Query: {query}")
print(f"Result: {documents[index]}")
print(f"Similarity: {score}")