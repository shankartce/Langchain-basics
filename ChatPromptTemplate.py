import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful technical assistant."),
    ("human", "Explain the concept of {concept}."),
])

chain = chat_prompt | llm

# Format the messages
messages = chat_prompt.format_messages(
    concept="neural networks"
)

result = chain.invoke(chat_prompt)

print(result.content)

# # Display formatted messages
# for msg in messages:
#     print(f"{msg.type.upper()}: {msg.content}")
