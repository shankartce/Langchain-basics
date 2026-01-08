import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


prompt = PromptTemplate(
    input_variables=["capital", "country"],
    template="What is the capital of {country}?"
)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

chain = prompt | model

response = chain.invoke(
    {
        "capital": "capital",
        "country": "India"
    }
)

print(response.content)
