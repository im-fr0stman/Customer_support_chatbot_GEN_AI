import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.chat_chain import get_conversational_chain

def run_chat():
    chain = get_conversational_chain()

    print("🔹 Ask something (type 'exit' to quit):")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        result = chain.invoke({"question": question})
        print("Bot:", result["answer"])

if __name__ == "__main__":
    run_chat()
