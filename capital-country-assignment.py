import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv["GEMINI_API_KEY"] 

prompt = PromptTemplate(
    input_variables=["topic"],
    template="What is the {capital} of the {country}?"
)




model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5
)

