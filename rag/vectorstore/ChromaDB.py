import os
import hashlib
import chromadb
from chromadb.config import Settings
from typing import List,Dict,Any,Tuple
import numpy as np
from pathlib import Path

## Anchored to this file so the store lands in RAG/database no matter
## which folder Python is started from
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parents[2] / "database")

class VectorStore:

    CONFIGURATION = {"hnsw" : {"space" : "cosine"}}
    METADATA = {"description": "Yt document embedding for RAG"}

    def __init__(self,collection_name:str = "yt_transcript",persist_directory:str = DEFAULT_PERSIST_DIR ):
        self.collection_name = collection_name
        self.persist_directory  = persist_directory
        self.collection = None
        self.client = None
        self._initialize_store()

    def _create_collection(self):
        """Create or get a collection in a cosine space"""
        return self.client.get_or_create_collection(
                name = self.collection_name,
                configuration = self.CONFIGURATION,
                metadata  = self.METADATA
        )

    def _space(self):
        """Distance space the collection was actually built with it"""
        try:
            return self.collection.configuration_json["hnsw"]["space"]
        except (KeyError, TypeError, AttributeError):
            return None
        
    def _initialize_store(self):
            """Initialize ChromaDB client and collection"""
            try:
                # Create persistent ChromaDB client
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_directory)
    
                # Get or create collection
                self.collection = self._create_collection()
    
                # A collection persisted by an earlier run may still be in l2 space.
                # Rebuild it so the similarity scores mean what the retriever thinks.
                if self._space() != "cosine":
                    print(f"Collection is in '{self._space()}' space, rebuilding in cosine space...")
                    self.client.delete_collection(name=self.collection_name)
                    self.collection = self._create_collection()
                    print("Rebuilt empty. Re-run the ingestion cell to repopulate it.")
    
                print(f"Vector store initialized. Collection: {self.collection_name} (space: {self._space()})")
                print(f"Existing documents in collection: {self.collection.count()}")
    
            except Exception as e:
                print(f"Error initializing vector store: {e}")
                raise
    def clear(self):
            """Drop the collection and recreate it empty.
    
            The store is persistent, so re-running ingestion would otherwise stack
            another copy of every chunk on top of what is already there.
            """
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._create_collection()
            print(f"Collection '{self.collection_name}' cleared. Documents: {self.collection.count()}")
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
            """
            Add documents and their embeddings to the vector store
    
            Args:
                documents: List of LangChain documents
                embeddings: Corresponding embeddings for the documents
            """
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
    
            print(f"Adding {len(documents)} documents to vector store...")
    
            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []
    
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Deterministic ID: re-ingesting the same chunk overwrites it
                # instead of inserting a duplicate under a fresh uuid.
                source = doc.metadata.get("source_file", "unknown")
                digest = hashlib.sha1(
                    f"{source}|{doc.page_content}".encode("utf-8")
                ).hexdigest()[:16]
                doc_id = f"doc_{digest}"
                ids.append(doc_id)
    
                # Prepare metadata
                metadata = dict(doc.metadata)
                metadata['doc_index'] = i
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)
    
                # Document content
                documents_text.append(doc.page_content)
    
                # Embedding
                embeddings_list.append(embedding.tolist())
    
            # Add to collection
            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text
                )
                print(f"Successfully added {len(documents)} documents to vector store")
                print(f"Total documents in collection: {self.collection.count()}")
    
            except Exception as e:
                print(f"Error adding documents to vector store: {e}")
                raise
    
