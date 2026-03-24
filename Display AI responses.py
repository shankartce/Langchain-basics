import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    input_variables=["language"],
    template="Write a short example program in {language}."
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

chain = prompt | llm | StrOutputParser()

formatted_prompt = prompt.format(language="Python")
response = chain.invoke(formatted_prompt)


print("AI Response:")
print(response)

