from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# model_name = "mixedbread-ai/mxbai-embed-large-v1"
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
# )
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = OllamaLLM(model="llama3.1")
embedding_model = OllamaEmbeddings(model="llama3.1")
# uploading the pdf
print("uploaded the pdf")

pdf_reader = PyPDFLoader("../Backend/assets/ipc.pdf")
documents = pdf_reader.load_and_split()
# pages = 'hello'
# print(pages)
chat_prompt = ChatPromptTemplate.from_messages([
    """
    <s>[INST] You are an AI legal assistant, skilled in Indian law. Provide accurate legal advice or document lookup based on the query. If you do not have sufficient information to answer, please advise on the next steps or suggest seeking professional legal help. [/INST]</s>
    [INST] Question: {query}
           Context: {context}
           Response:
    [/INST]
"""
])
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
def handle_query(query, context=""):
    # If document search is triggered
    if "section" in query.lower() or "pdf" in query.lower():
        results = vector_store.similarity_search(query)
        if results:
            context = results[0].page_content  # Add the first result to the context

    # Construct the full prompt with context and query
    full_prompt = chat_prompt.format(query=query, context=context)
    return llm.invoke(full_prompt)

# Example usage
chat_history = []

while True:
    user_query = input("Ask your question: ")
    if user_query.lower() == "exit":
        break
    
    # Add previous conversation to context
    context = " ".join(chat_history)
    
    # Handle the query
    response = handle_query(user_query, context)
    
    # Update chat history
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"AI: {response}")
    
    print(response)