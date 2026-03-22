from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

class Topic(BaseModel):
    description: str = Field(description="A brief explanation of the topic")
    hashtags: list[str] = Field(description="Related keywords as hashtags")

parser = JsonOutputParser(pydantic_object=Topic)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer concisely."),
        ("user", "#FORMAT:\n{format_instructions}\n\n#QUESTION:\n{question}")
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | StrOutputParser()

result = chain.invoke(
    {"question": "Explain the seriousness of global warming."}
)

structured_output = parser.parse(result)

print(structured_output)
