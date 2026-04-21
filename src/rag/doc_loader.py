import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

def load_docs():
    pdf_directory = "docs/pdfs"
    text_directory = "docs/texts"

    pdf_loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    text_loader = DirectoryLoader(
        text_directory,
        glob="**/*.txt",
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    )
    documents = []

    if os.path.exists(pdf_directory):
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
    else: 
        print(f"PDF directory '{pdf_directory}' does not exist.")
    
    if os.path.exists(text_directory):
        text_docs = text_loader.load()
        documents.extend(text_docs)
    else:
        print(f"Text directory '{text_directory}' does not exist.")
    
    #print(f"Loaded {len(documents)} documents.")
    #print(f"Sample document metadata: {documents[0].metadata if documents else 'No documents loaded.'}")
    #print(f"Sample document content: {documents[0].page_content[:200]}...") if documents else print("No documents loaded.")
    return documents

if __name__ == "__main__":
    load_docs()