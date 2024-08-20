from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm = ChatOllama(model="llama3.1",temperature=0.7,)
embedding_model = OllamaEmbeddings(model="llama3.1")
chat_history=[]
# uploading the pdf
print("uploaded the pdf")

pdf_reader = PyPDFLoader("../Backend/assets/ipc.pdf")
documents = pdf_reader.load_and_split()


chat_prompt = PromptTemplate.from_template(
        """
        You are an AI Legal Assistant Skilled in Indian Law ,Also you are a very Friendly Chat Bot .
        So for each user query provide ACCURATE,USEFUL,THOUGHTFUL Response.
        also if the user intends to do normal Chatting ; initiate in friendly chatting too,
        But Remind them of your PURPOSE and ROLE if the user initiates only FRIENDLY CHAT and make them engage in Asking LEGAL QUERIES
        on Indian Peanal Code
        Question: {input}
        Context: {context}
        Response:
    """
)
retriever_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        (
            "human",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
        ),
    ]
)
print("creating the text splitter")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=200, length_function=len
)
# creating the chunks
print("creating the chunks")
chunks = text_splitter.split_documents(documents)
print("creating chroma db")

vector_store = Chroma(persist_directory="db", embedding_function=embedding_model)

print("Chroma DB loaded successfully")


while(1):
    query=input("prompt >> ")
    