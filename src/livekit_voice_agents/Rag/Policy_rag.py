from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "docs/Company-Policy-and-Procedure.pdf"

# load the documents in pdf formate 
Loader = PyPDFLoader(pdf_path)


docs = Loader.load()

# split the documents into chunks 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


