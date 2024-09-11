from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from sklearn.metrics.pairwise import cosine_similarity
llm = ChatOllama(model="llama3.1",temperature=0.7,)
embedding_model = OllamaEmbeddings(model="llama3.1")
# uploading the pdf
print("uploaded the pdf")
# memory = ChatMessageHistory(memory_key="chat_history", return_messages=True)

pdf_reader = PyPDFLoader("../Backend/assets/ipc.pdf")
documents = pdf_reader.load_and_split()




chat_prompt = PromptTemplate.from_template(
        """
        You are an AI Legal Assistant Skilled in Indian Law,Your name is LexAi ,Also you are a very Friendly Chat Bot .
        So for each user query provide ACCURATE,USEFUL,THOUGHTFUL Response.
        also if the user intends to do normal Chatting ; initiate in friendly chatting too,
        But Remind them of your PURPOSE and ROLE if the user initiates only FRIENDLY CHAT and compel them to engage in Asking LEGAL QUERIES
        on Indian Peanal Code
        Question: {input},
        Context: {context},
        Response:
    """
)
# chat_prompt = PromptTemplate.from_template(
#         """
#         Your name is LexAi,You are an AI Legal Consultant Skilled in Indian Law ,Also you are a very Friendly Chat Bot .
#         for each user query provide ACCURATE,USEFUL,THOUGHTFUL Response.
#         if the user is engaging in casual chats REMIND THEM ABOUT YOUR ROLE AND PURPOSE AS AN AI LEGAL CONSULTANT
#         Question: {input},
#         Context: {context},
#         Response:
#     """
# )

legal_check_template = ChatPromptTemplate.from_template(
    """
    You are an AI Legal Assistant. A legal question typically contains terms related to laws, sections, or legal topics.
    A query like 'Can you explain Section 420 of IPC?' is a legal query.
    """
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




examples = {
    "legal": [
        "What is Section 420 of the Indian Penal Code?",
        "Can you explain the punishment for theft under IPC?",
        # "Tell me about bail provisions in Indian law.",
        # "What are the laws regarding domestic violence?",
        # "How do I file a complaint for defamation?",
        # "What is the penalty for cybercrime in India?",
        # "Explain the law on trespassing in India.",
        # "What are the conditions for anticipatory bail?",
        # "What is the difference between murder and manslaughter?",
        # "Tell me about the legal rights of tenants in India.",
        # "How does Indian law handle intellectual property theft?",
        # "What does the Indian Penal Code say about kidnapping?",
        # "What is the punishment for tax evasion?",
        # "Can you explain the law on child custody?",
        # "What are the rights of employees under Indian labor law?",
        # "How is dowry harassment handled under IPC?",
        # "What are the grounds for divorce in India?",
        # "Tell me about the provisions for medical negligence.",
        # "What are the legal implications of a breach of contract?",
        # "How can someone be arrested without a warrant in India?",
        # "Explain the legal procedures for filing an FIR.",
        # "What are the penalties for drug possession in India?",
        # "Can a minor be tried for a criminal offense in India?",
        # "What is the punishment for money laundering under Indian law?",
        # "How is evidence handled in Indian courts?",
    ],
    "casual": [
        "How's the weather today?",
        "Tell me a joke.",
        "What's your name?",
        # "How are you doing today?",
        # "What's the time now?",
        # "What's your favorite color?",
        # "Who is your favorite actor?",
        # "Can you recommend a good book?",
        # "What's your favorite hobby?",
        # "Do you like music?",
        # "Tell me about your favorite movie.",
        # "What's your favorite food?",
        # "Do you have any pets?",
        # "What's the best place to travel?",
        # "What's your favorite sport?",
        # "How do you spend your weekends?",
        # "What kind of music do you enjoy?",
        # "Do you like reading?",
        # "What are your hobbies?",
        # "What's the best vacation you've had?",
        # "Do you have any siblings?",
        # "Can you suggest a good restaurant nearby?",
        # "What's your opinion on the latest movie release?",
        # "How do you relax after a long day?",
        # "What's your favorite TV show?",
    ]
}

threshold = 1.4

# Embed all legal and casual sentences from the dictionary
legal_embeddings = embedding_model.embed_documents(examples["legal"])
casual_embeddings = embedding_model.embed_documents(examples["casual"])

# Function to calculate similarity with the dictionary
def classify_query(query):
    # Embed the query
    query_embedding = embedding_model.embed_documents([query])

    # Calculate similarities
    legal_similarity = cosine_similarity(query_embedding, legal_embeddings)
    casual_similarity = cosine_similarity(query_embedding, casual_embeddings)

    # Get the maximum similarity score for legal and casual categories
    max_legal_similarity = max(legal_similarity[0])
    max_casual_similarity = max(casual_similarity[0])

    # Set a threshold for classifying the query
    threshold = 0.4

    # Compare similarities to classify as legal or casual
    if max_legal_similarity > threshold and max_legal_similarity > max_casual_similarity:
        return {"status":"legal","score":max_legal_similarity}
    else:
        return {"status":"casual","score":max_casual_similarity}

print("sample check point")

chat_context = []

def handle_query(query):

    chat_context.append(("human",query))

    query_type = classify_query(query)

    if query_type["status"] == "casual":
        # If casual, handle the query normally with LLM
        context = 'friendly chat '
        casual_prompt = chat_prompt.format(input=query,context=chat_context)
        print(chat_context,"\n")
        docs = llm.invoke(casual_prompt)
        chat_context.append(("ai",docs.content))
    else:
        # If legal, perform similarity search in Chroma for relevant docs
        relevant_docs = vector_store.similarity_search_with_score(query, k=10)
        context = "".join([doc[0].page_content for doc in relevant_docs])
        print(chat_context,"\n")
        full_prompt = chat_prompt.format(input=query, context=context)
        docs = llm.invoke(full_prompt)
        chat_context.append(("ai",docs.content))

    return docs.content
    


while True:
    query = input("Ask your question: ")
    if query.lower() == "bye":
        break
    response = handle_query(query)
    print("\nAI Response:\n")
    print(response)
    print("\n------------------------------------\n")
