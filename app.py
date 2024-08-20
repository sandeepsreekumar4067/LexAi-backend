from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama3.1",
)
#uploading the pdf 
print("uploaded the pdf")

pdf_reader = PyPDFLoader('../Backend/assets/ipc.pdf')
pages = pdf_reader.load_and_split()
# print(pages)

print("creating the text splitter")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)


#splitting the ipc
print("splitted chunks has been created")
chunks = text_splitter.split_documents(pages)

print("Embedding the chunks")
embedded_chunks = embeddings.embed_documents([chunk.page_content for chunk in chunks])
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:")
#     print(chunk.page_content)  # `page_content` is the text in this chunk
#     print("-----")