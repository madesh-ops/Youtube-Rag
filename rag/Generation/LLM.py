import os
from typing import List,Dict,Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.retrieval import RAGRetriever

load_dotenv()

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a YouTube video using only the transcript excerpts below. "
     "If the answer is not in the excerpts, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

class RAGGenerator:

    def __init__(self,retriever:RAGRetriever,model_name:str = "openai/gpt-oss-120b",temperature:float = 0.1):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set. Add it to a .env file or your environment.")
        self.retriever = retriever
        self.llm = ChatGroq(model=model_name,temperature=temperature)
        self.chain = PROMPT | self.llm | StrOutputParser()

    def _build_context(self,docs:List[Dict[str,Any]]) -> str:
        return "\n\n".join(f"[{doc['rank']}] {doc['content']}" for doc in docs)

    def answer(self,question:str,top_k:int = 5,score_threshold:float = 0.0) -> Dict[str,Any]:
        docs = self.retriever.retrieve(question,top_k=top_k,score_threshold=score_threshold)
        if not docs:
            return {"answer": "No relevant context found in the transcript.", "sources": []}

        answer = self.chain.invoke({"context": self._build_context(docs), "question": question})
        return {"answer": answer, "sources": docs}
