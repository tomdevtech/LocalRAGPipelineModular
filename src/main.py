from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a perfect summarizer for restaurent reviews.

The details you can find here: {reviews}

And here is the question to the answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n")
    question = input("Ask your question (or type 'q' to quit): ")
    if question.lower() == 'q':
        break

    print("\n\n")
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)