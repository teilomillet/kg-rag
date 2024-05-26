# query.py

import fire
import kuzu
from config import Config
from loguru import logger
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.mistralai import MistralAIEmbedding


class KnowledgeGraphQuery:
    def __init__(self, db_name="default_db"):
        """
        Initialize the knowledge graph query engine with a specific database.
        """
        logger.info(f"Connecting to database: {db_name}")
        self.db = kuzu.Database(db_name)
        self.graph_store = KuzuGraphStore(self.db)
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )

        # Set up the Groq LLM with API key from the configuration
        embed = MistralAIEmbedding(
            model_name="mistral-embed", api_key=Config.MISTRAL_API_KEY
        )
        Settings.embed_model = embed
        Settings.chunk_size = 512
        Settings.llm = Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)

        # Assuming that we are loading an existing index
        self.index = KnowledgeGraphIndex(nodes=[], storage_context=self.storage_context)
        logger.debug("Knowledge graph index initialized.")

    def query_graph(self):
        """
        Query the knowledge graph interactively with user input.
        """
        query_text = input("Enter your query: ")  # User inputs their query
        query_engine = self.index.as_query_engine(
            include_text=True, response_mode="tree_summarize"
        )
        logger.debug(f"Querying with: {query_text}")
        response = query_engine.query(query_text)
        logger.info("Query executed successfully.")
        print("Response:\n", response)


def main(db_name="default_db"):
    """
    Main function to run the knowledge graph query engine interactively.
    Allow database name to be specified dynamically.
    """
    logger.add("query.log", rotation="1 week")  # Save logs to a file
    kg_query = KnowledgeGraphQuery(db_name)
    kg_query.query_graph()


if __name__ == "__main__":
    fire.Fire(main)
