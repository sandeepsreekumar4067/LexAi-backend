from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory

llm = ChatOllama(model="llama3.1",temperature=0.7,)
embedding_model = OllamaEmbeddings(model="llama3.1")
chat_history=[]
# uploading the pdf
print("uploaded the pdf")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

pdf_reader = PyPDFLoader("../Backend/assets/ipc.pdf")
documents = pdf_reader.load_and_split()


chat_prompt = PromptTemplate.from_template(
        """
        You are an AI Legal Assistant Skilled in Indian Law,Your name is LexAi ,Also you are a very Friendly Chat Bot .
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

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 20, "score_threshold": 0.1},
)
print("document retirver initialised ")
history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=retriever_prompt
)

print("memmory history created...")

document_chain = create_stuff_documents_chain(llm, chat_prompt)
print("document chaining...")

retrieval_chain = create_retrieval_chain(
    history_aware_retriever,
    document_chain,
)

print("retirever chain created....")

def handle_query(query):
    result = retrieval_chain.invoke({"input": query})
    return result["answer"]
chat_history = []

while True:
    query = input("Ask your question: ")
    if query.lower() == "exit":
        break
    response = handle_query(query)
    print("\nAI Response:\n")
    print(response)
    print("\n------------------------------------\n")
