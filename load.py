# load.py

import fire
from config import Config
from llama_index.core import (
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    Settings,
    StorageContext,
)
from llama_index.llms.groq import Groq
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from kuzu import Database
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.embeddings.mistralai import MistralAIEmbedding
from loguru import logger


class DocumentIndexer:
    def __init__(self, db_name="default_db"):
        logger.info(f"Initializing database connection: {db_name}")
        self.db = Database(db_name)
        self.graph_store = KuzuGraphStore(self.db)
        logger.debug("Database and graph store initialized successfully.")

    def build_pipeline(self):
        logger.info("Building the document ingestion pipeline.")
        llm = Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)
        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
            SummaryExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
        ]
        return IngestionPipeline(transformations=transformations)

    def index_documents(self, directory_path, append=False):
        logger.info(f"Loading documents from {directory_path}")
        documents = SimpleDirectoryReader(directory_path).load_data()
        if not documents:
            logger.warning(
                "No documents loaded. Check the directory path and data format."
            )
        else:
            logger.info(f"Loaded {len(documents)} documents")

        embed = MistralAIEmbedding(
            model_name="mistral-embed", api_key=Config.MISTRAL_API_KEY
        )
        Settings.embed_model = embed
        Settings.chunk_size = 512
        Settings.llm = Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)
        pipeline = self.build_pipeline()
        processed_documents = pipeline.run(documents)

        logger.info("Documents processed, initializing knowledge graph index.")
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)

        if append:
            index = KnowledgeGraphIndex(nodes=[], storage_context=storage_context)
        else:
            index = KnowledgeGraphIndex.from_documents(
                processed_documents,
                max_triplets_per_chunk=2,
                storage_context=storage_context,
            )

        # Assuming 'commit' or equivalent operation is necessary here
        # index.commit()  # Uncomment if such a method exists
        logger.success("Indexing complete.")


def main(*args):
    logger.add("document_indexing.log", rotation="1 week")  # Save logs to a file
    if len(args) == 1:
        directory_path = args[0]
        db_name = "default_db"
        append = False
    elif len(args) == 2:
        db_name = args[0]
        directory_path = args[1]
        append = True
    else:
        logger.error("Incorrect number of arguments provided.")
        raise ValueError("Usage: python script_name.py [db_name] dir_path")

    indexer = DocumentIndexer(db_name)
    indexer.index_documents(directory_path, append)
    logger.info(
        f"Documents from {directory_path} have been processed in database '{db_name}'."
    )


if __name__ == "__main__":
    fire.Fire(main)
