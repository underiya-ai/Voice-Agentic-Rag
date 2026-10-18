from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharracterTextSplitter


PDF_PATH = Path("docs/Company-policy-and-procedure.pdf")

def main():

    # load pdf 
    docs = PyPDFLoader(str(PDF_PATH)).load()

    # con