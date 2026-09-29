from src.graph import graph

questions = [
    "What is Agentic AI?",
    "What are the characteristics of Agentic AI?",
    "How does Agentic AI make decisions?",
    "What are the benefits of Agentic AI?",
    "What is the role of memory in Agentic AI?",
    "Who won the FIFA World Cup in 2022?"
]

for question in questions:
    print("\nQuestion:", question)

    result = graph.invoke({
        "question": question
    })

    print("Answer:", result["answer"])