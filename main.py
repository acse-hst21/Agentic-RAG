from dotenv import load_dotenv
load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("---Advanced RAG Implementation---")
    print(app.invoke(input={"question": "who won the champions league in 2010?"}))