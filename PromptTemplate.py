import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# prompt = PromptTemplate(
#     input_variables=["topic", "audience"],
#     template="Explain {topic} in simple terms for a {audience}."
# )

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

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)
# formatted_prompt = prompt.format(
#     question="What did Stephen Hawking discover?"
# )

# print(formatted_prompt)

chain = prompt | llm | StrOutputParser()

# result = chain.invoke(formatted_prompt)

for chunk in chain.stream({"question": "What are the main responsibilities of a prime minister?"}):
    print(chunk)

# response = llm.invoke(formatted_prompt)
# response_metadata = getattr(response, "response_metadata", {})
# print(response_metadata)

# print("AI Response:")
# print(response.content)

# print(result)

# def shout(text: str) -> str:
#     return text.upper()

# chain = prompt | model | StrOutputParser() | shout

# result = chain.invoke({"topic": "dogs"})
# print(result)
