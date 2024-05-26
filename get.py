# get.py

import fire
import kuzu

from config import Config
from langchain_groq import ChatGroq

from langchain.chains import KuzuQAChain
from langchain_community.graphs import KuzuGraph


class QueryHandler:
    def __init__(self):
        # Initialize the database and graph store
        self.db = kuzu.Database("test1")
        self.graph = KuzuGraph(self.db)
        self.chain = KuzuQAChain.from_llm(
            ChatGroq(model="llama3-70b-8192", groq_api_key=Config.GROQ_API_KEY),
            graph=self.graph,
            verbose=True,
        )

    def query(self):
        print("Enter your query or type 'exit' to stop:")
        while True:
            user_input = input("Your query: ")
            if user_input.lower() in {"exit", "quit", ":q", "q"}:
                print("Exiting query handler.")
                break  # Exit the loop if the user types 'exit'

            try:
                # Process the query and print the result
                response = self.chain.invoke(user_input)
                print(f"Query result: {response}")
            except Exception as e:
                # Handle any exceptions that may occur during query processing
                print(f"An error occurred during query processing: {e}")


def main():
    fire.Fire(QueryHandler().query)


if __name__ == "__main__":
    main()
