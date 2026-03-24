from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

class ReviewAnalysis(BaseModel):
    sentiment: str =Field(description="Overall Sentiment: POSITIVE, NEGATIVE")
    rating: int = Field(description="Estimated rating out of 10")
    summary: str = Field(description="One-sentence summary")
    recommended: bool = Field(description="True or False")

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt = PromptTemplate(
    template="""
    You are a movie review analyst.
    Analyze the given review and extract strucutured in information:
    Here are some  examples:
    Review: "Absolutely stunning visuals and a gripping storytelling"
    Analysis: sentiment:NEGATIVE, rating=7, recommended=False, summary="Reviewer said that this movie is absoluting stunning"

    Now analyze the following review:
    review: {review}
    {format_instructions}

    """,
    input_variables=["review"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

chain = prompt | llm | parser

test_review = "The movie had some great action sequences but the ending was disapponting. I'd say it's worth watching once but not twice."

result = chain.invoke({"review": test_review})


print(f"Sentiment {result.sentiment}")
print(f"Rating: {result.rating}/10")
print(f"Summary: {result.summary}")
print(f"Recommended: {result:recommended}")







