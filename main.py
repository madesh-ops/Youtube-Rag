from rag.Ingestion import YTSloader ,Chunking
from rag.Embedding import EmbeddingManager
from rag.vectorstore import VectorStore
from rag.retrieval import RAGRetriever
from rag.Generation import RAGGenerator

video_id = input("Enter the video id : ")

### example_id : qbLc5a9jdXo

Transcript = YTSloader(video_id,"en")
chunks = Chunking(Transcript)


### initializing classes


embedding_manager = EmbeddingManager()
vectorstore=VectorStore()
rag_retriever = RAGRetriever(vector_store=vectorstore, embedding_manager=embedding_manager)

### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

## Generate the Embeddings

embeddings=embedding_manager.generate_embeddings(texts)

## Wipe what is already persisted so the store holds exactly one copy of
## each chunk (the earlier runs left duplicates behind).
vectorstore.clear()

## store in the vector database
vectorstore.add_documents(chunks,embeddings)

## Answer the question with the Groq LLM, using the retrieved chunks as context
generator = RAGGenerator(rag_retriever)

question = input("Ask a question about the video : ")
result = generator.answer(question)
print(result["answer"])