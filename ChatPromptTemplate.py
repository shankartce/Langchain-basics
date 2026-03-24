import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful technical assistant."),
    ("human", "Explain the concept of {concept}."),
])

chain = chat_prompt | llm | StrOutputParser

# Format the messages
messages = chat_prompt.format_messages(
    concept="neural networks"
)

result = chain.invoke({"concept": "neural networks"})

print(result.content)

# Display formatted messages
for msg in messages:
    print(f"{msg.type.upper()}: {msg.content}")

# Instruction, Context, Input, Output