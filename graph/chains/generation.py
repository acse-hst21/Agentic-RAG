from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks."),
    ("human", "{context}\n\nQuestion: {question}")
])

llm = ChatOpenAI(temperature=0)
generation_chain = prompt | llm | StrOutputParser()
