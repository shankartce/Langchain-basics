import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

prompt_title = "summary-stuff-documents"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

prompt_template = """Please summarize the content according to the following rules.


REQUEST:
1. Summarize the main points in bullet points.
2. Each bullet must start with an emoji that matches its meaning.
3. Use varied emojis.
4. Do NOT include unnecessary information.
5. Write the summary in English.

CONTEXT:
{context}

SUMMARY:
"""
prompt = PromptTemplate.from_template(prompt_template)

formatted_prompt = prompt.format(
    context="""
    Cats are fascinating and highly adaptable animals known for their independence, agility, and subtle intelligence.
    As descendants of solitary hunters, they retain strong instincts—sharp reflexes, keen night vision, and an ability to move silently—which make them excellent predators even in domestic environments. 
    At the same time, cats form complex social bonds with humans, often expressing affection through behaviors like purring, head-butting, and following their owners from room to room. 
    Their personalities can vary widely, from playful and energetic to calm and observant, making each cat uniquely engaging as a companion.
    """
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke(formatted_prompt)

print(result)
