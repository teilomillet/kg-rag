# get.py

import fire
import kuzu

from config import Config
from llama_index.llms.groq import Groq
from llama_index.embeddings.mistralai import MistralAIEmbedding

from llama_index.core import Settings, KnowledgeGraphIndex
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.core import StorageContext


class QueryHandler:
    def __init__(self):
        # Initialize the database and graph store
        self.db = kuzu.Database("test1")
        self.graph = KuzuGraphStore(self.db)
        Settings.llm = Groq(model="llama3-70b-8192", groq_api_key=Config.GROQ_API_KEY)
        Settings.embed_model = MistralAIEmbedding(
            "mistral-embed", api_key=Config.MISTRAL_API_KEY
        )
        storage_context = StorageContext.from_defaults(graph_store=self.graph)
        index = KnowledgeGraphIndex(nodes=[], storage_context=storage_context)
        self.query_engine = index.as_query_engine(
            include_text=False, response_mode="tree_summarize"
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
                response = self.query_engine.query(user_input)
                print(f"Query result: {response}")
            except Exception as e:
                # Handle any exceptions that may occur during query processing
                print(f"An error occurred during query processing: {e}")


def main():
    fire.Fire(QueryHandler().query)


if __name__ == "__main__":
    main()
