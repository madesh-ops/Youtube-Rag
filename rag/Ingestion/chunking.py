from langchain_text_splitters import RecursiveCharacterTextSplitter

def Chunking(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators=["\n\n","\n"," ",""],
        length_function = len 
    )

    split_docs = text_splitter.split_documents(documents)
    return split_docs