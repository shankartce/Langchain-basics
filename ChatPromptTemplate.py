from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful technical assistant."),
    ("human", "Explain the concept of {concept}."),
])

# Format the messages
messages = chat_prompt.format_messages(
    concept="neural networks"
)

# Display formatted messages
for msg in messages:
    print(f"{msg.type.upper()}: {msg.content}")
