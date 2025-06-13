from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama3-70b-8192")

# output Schema
class outputFormatter(TypedDict):
    key_themes : Annotated[list[str], "Write down all the key themes discussed in the review in a list."]
    summary : Annotated[str, "Write down the short summary of the review."]
    sentiment : Annotated[Literal["positive", "negative", "neutral"], "Write down the sentiment of the review."]
    rating : Annotated[Optional[int], "Write down the rating of the review. If not available, return None."]
    pros : Annotated[Optional[list[str]], "Write down the pros of the review. If not available, return None."]
    cons : Annotated[Optional[list[str]], "Write down the cons of the review. If not available, return None."]

structured_model = model.with_structured_output(outputFormatter)

Review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
"""

result = structured_model.invoke(Review)

print(result)