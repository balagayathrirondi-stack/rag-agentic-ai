from src.graph import graph

print("RAG Chatbot started!")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = graph.invoke({"question": question})

    print("AI:", result["answer"])
    print()