from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# model_name = "mixedbread-ai/mxbai-embed-large-v1"
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
# )
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = OllamaEmbeddings(model="llama3.1")
# uploading the pdf
print("uploaded the pdf")

pdf_reader = PyPDFLoader("../Backend/assets/ipc.pdf")
documents = pdf_reader.load_and_split()
# pages = 'hello'
# print(pages)

print("creating the text splitter")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=200, length_function=len
)
# creating the chunks
print("creating the chunks")
chunks = text_splitter.split_documents(documents)
print("creating chroma db")
# creating the vector store
# vector_store = Chroma.from_documents(
#     chunks, embedding_model, persist_directory="db"
# )
# Load the existing Chroma vector store
# vector_store = Chroma(persist_directory="db", embedding_function=embedding_model.embed_query)
vector_store = Chroma(persist_directory="db", embedding_function=embedding_model)


print("Chroma DB loaded successfully")
# vector_store.persist()
while(1):
    query = input(">>")
    results = vector_store.similarity_search(query)
    for result in results:
        print(result.page_content)