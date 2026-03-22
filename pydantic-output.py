from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

email_conversation = """
From: Alex Morgan (alex.morgan@bikesupply.io)
To: Emma Lindholm (emma@northcycle.fi)
Subject: ZENESIS bicycle distribution collaboration

Hello Emma,

We recently learned about your company through a press release.
We are interested in distributing the ZENESIS electric bicycle.

Could you share a detailed brochure covering specifications, battery performance,
and design highlights?

I would also like to propose a meeting next Tuesday at 10:00 AM
to discuss collaboration opportunities at your office.

Best regards,
Alex Morgan
Managing Director
Bike Supply Co.
""" 

class EmailSummary(BaseModel):
    sender_name: str = Field(description="Name of the email sender")
    sender_email: str = Field(description="Email address of the sender")
    subject: str = Field(description="Email subject line")
    summary: str = Field(description="Short summary of the email body")
    meeting_time: str = Field(description="Proposed meeting date and time")

parser = PydanticOutputParser(pydantic_object=EmailSummary)

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant.

TASK:
Extract the key information from the email below.

EMAIL:
{email}

FORMAT:
{format}
"""
).partial(format=parser.get_format_instructions())

chain = prompt | llm | StrOutputParser()

raw_output = chain.invoke({"email": email_conversation})

structured_output = parser.parse(raw_output)

print(structured_output)
