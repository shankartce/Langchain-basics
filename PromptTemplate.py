import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Explain {topic} in simple terms for a {audience}."
)


formatted_prompt = prompt.format(
    topic="machine learning",
    audience="high school student"
)

print(formatted_prompt)
