# upload.py

import fire
import kuzu
from loguru import logger

from config import Config
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    KnowledgeGraphIndex,
    Settings,
)
from llama_index.llms import groq, mistralai
from llama_index.embeddings.mistralai import MistralAIEmbedding


class DirectoryUploader:
    def __init__(self):
        logger.debug("Initializing the Database...")
        self.db = kuzu.Database("test1")
        self.graph_store = KuzuGraphStore(self.db)

        logger.debug("Initializing LLMs...")
        self.llm = groq.Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)
        self.embed = MistralAIEmbedding(
            model_name="mistral-embed", api_key=Config.MISTRAL_API_KEY
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed
        Settings.chunk_size = 1536
        logger.debug("Settings configured.")

    def upload(self, directory_path):
        logger.debug(f"Starting the upload process for {directory_path}...")
        try:
            documents = SimpleDirectoryReader(directory_path).load_data()
            logger.debug(f"Loaded documents: {documents}")

            storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            logger.debug("Storage context created. Next step can take some time...")

            # Can take a while
            KnowledgeGraphIndex.from_documents(
                documents, max_triplets_per_chunk=2, storage_context=storage_context
            )
            logger.debug("Documents have been indexed.")

            print(
                f"Documents from {directory_path} have been indexed in the knowledge graph."
            )
        except Exception as e:
            logger.error(f"An error occurred during indexing: {e}")
            print(f"An error occurred during indexing: {e}")


def main():
    fire.Fire(DirectoryUploader().upload)


if __name__ == "__main__":
    main()
