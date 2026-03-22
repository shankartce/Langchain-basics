from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

examples = [
    {
        "question": "Who lived longer, Steve Jobs or Albert Einstein?",
        "answer": """Do we need additional questions? Yes.
Steve Jobs died at age 56.
Albert Einstein died at age 76.
Final answer: Albert Einstein."""
    },
    {
        "question": "When was Google's founder born?",
        "answer": """Do we need additional questions? Yes.
Google was founded by Larry Page.
Larry Page was born on March 26, 1973.
Final answer: March 26, 1973."""
    },
]

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "instruction": "Please write meeting minutes",
    "input": "On December 26, the product team reviewed project progress..."
})

print(response.content)
