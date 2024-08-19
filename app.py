from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Step 1: Load Your Legal PDFs
print("ipc detected")
ipc_loader = PyPDFLoader("../Backend/assets/ipc.pdf")  # Replace with your IPC PDF path
constitution_loader = PyPDFLoader("../Backend/assets/constitution.pdf")  # Replace with your Constitution PDF path

# Load documents from PDFs
print("ipc loading..")

ipc_docs = ipc_loader.load()
constitution_docs = constitution_loader.load()

# Combine the documents into a single list
documents = ipc_docs + constitution_docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk_document = text_splitter.split_documents(documents)
print(chunk_document)
print("creating embedding model")

# Step 2: Initialize the Ollama Embeddings Model with Llama 3.1
embedding_model = OllamaEmbeddings(
    model="llama3.1",
)
print("embedding complete")

# Step 3: Create a Vector Store (using FAISS for efficient retrieval)
print("vectore store creating")

vector_store = FAISS.from_documents(chunk_document, embedding_model)
query="what is a petty case ?"
res=vector_store.similarity_search(query)
print(res[0].page_content)

# # Step 4: Set up the Retrieval-based QA Chain
# qa_chain = RetrievalQA(llm=embedding_model, retriever=vector_store.as_retriever())

# # Function to ask a question and get an answer
# print("model ready")
# def ask_question(query):
#     response = qa_chain.run(query)
#     return response

# # Example usage: Querying the legal assistant
# if __name__ == "__main__":
#     while True:
#         user_query = input("Enter your legal question: ")
#         if user_query.lower() == "exit":
#             break
#         answer = ask_question(user_query)
#         print(f"Answer: {answer}")
