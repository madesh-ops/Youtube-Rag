from sentence_transformers import SentenceTransformer
import hashlib
import chromadb
import numpy as np
from chromadb.config import Settings
from typing import List,Dict,Any,Tuple

class EmbeddingManager:
    def __init__(self,model_name:str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model {self.model_name} ..... ")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Successfully Loaded {self.model_name}")
        except Exception as e:
            print(f"error occurred : {e} ")

    def generate_embeddings(self,text:list[str]) ->np.ndarray:

        if not self.model:
            raise ValueError("Model not loaded")
        print("Converting Chunks into Embeddings ... ")
        try:
            embeddings = self.model.encode(text)
        except Exception as e :
            print(f"Error Occured : {e}")
            raise

        return embeddings
               

