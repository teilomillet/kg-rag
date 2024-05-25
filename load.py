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


class DocumentIndexer:
    def __init__(self, db_name="default_db"):
        """
        Initialize the document indexer with a specific database.

        :param db_name: Name of the database to connect to.
        """
        self.db = Database(db_name)
        self.graph_store = KuzuGraphStore(self.db)

    def build_pipeline(self):
        """
        Build the ingestion pipeline with transformations using Groq model.

        :param llm: Groq language model instance.
        :return: Configured ingestion pipeline.
        """
        llm = Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)
        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
            SummaryExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
        ]
        return IngestionPipeline(transformations=transformations)

    def index_documents(self, directory_path, append=False):
        """
        Load documents from a directory and index them into the graph database.

        :param directory_path: Path to the directory containing documents to index.
        :param append: Flag to append to the existing database or create new entries.
        """
        # Load the documents
        documents = SimpleDirectoryReader(directory_path).load_data()

        # Define the LLM
        embed = MistralAIEmbedding(
            model_name="mistral-embed", api_key=Config.MISTRAL_API_KEY
        )
        Settings.embed_model = embed
        Settings.chunk_size = 512
        Settings.llm = Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)
        pipeline = self.build_pipeline()
        processed_documents = pipeline.run(documents)

        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)

        # Initialize the index and choose the update mode
        if append:
            index = KnowledgeGraphIndex(nodes=[], storage_context=storage_context)
        else:
            index = KnowledgeGraphIndex.from_documents(
                processed_documents,
                max_triplets_per_chunk=2,
                storage_context=storage_context,
            )

        # Commit changes to the graph database
        index  # Assuming commit() is the method to finalize changes
        print("Indexing complete.")


def main(*args):
    """
    Main function to run the document indexer.
    Usage:
      python script_name.py [db_name] dir_path

    If db_name is provided, append to the given database. If not, create a new one.
    """
    if len(args) == 1:
        # Assume default DB and new index
        directory_path = args[0]
        db_name = "default_db"
        append = False
    elif len(args) == 2:
        # Provided DB name and path
        db_name = args[0]
        directory_path = args[1]
        append = True
    else:
        raise ValueError(
            "Incorrect number of arguments provided. Usage: python script_name.py [db_name] dir_path"
        )

    indexer = DocumentIndexer(db_name)
    indexer.index_documents(directory_path, append)
    print(
        f"Documents from {directory_path} have been processed in database '{db_name}'."
    )


if __name__ == "__main__":
    fire.Fire(main)
