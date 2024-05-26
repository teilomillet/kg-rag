# upload.py

import os
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
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms import groq, mistralai
from llama_index.embeddings.mistralai import MistralAIEmbedding

class DocumentUploader:
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

    def upload(self, input):
        try:
            if os.path.isdir(input):
                # Input is a directory
                logger.debug(f"Loading documents from directory: {input}")
                documents = SimpleDirectoryReader(input).load_data()
            else:
                # Assume input is a Wikipedia page title
                logger.debug(f"Loading data from Wikipedia for: {input}")
                wiki_loader = WikipediaReader()
                documents = wiki_loader.load_data(pages=[input], auto_suggest=False)

            logger.debug(f"Loaded documents: {documents}")
            storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            logger.debug("Storage context created. Next step can take some time...")

            # Index documents in the knowledge graph
            KnowledgeGraphIndex.from_documents(
                documents, max_triplets_per_chunk=10, storage_context=storage_context,include_embeddings=True,
            )
            logger.debug("Documents have been indexed.")

            print(
                f"Documents from '{input}' have been indexed in the knowledge graph."
            )
        except Exception as e:
            logger.error(f"An error occurred during indexing: {e}")
            print(f"An error occurred during indexing: {e}")

def main():
    fire.Fire(DocumentUploader().upload)

if __name__ == "__main__":
    main()

