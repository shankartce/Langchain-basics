import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


prompt = PromptTemplate(
    input_variables=["language"],
    template="Write a short example program in {language}."
)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5
)


formatted_prompt = prompt.format(language="Python")
response = llm.invoke(formatted_prompt)


print("AI Response:")
print(response.content)

