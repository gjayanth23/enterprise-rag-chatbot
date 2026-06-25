from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("data/sample_policy.txt")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./vectorstore"
)

db.persist()

print("Vector database created successfully")
