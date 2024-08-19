from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from flask_cors import CORS
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
# from langchain.embeddings import LlamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
app = Flask(__name__)
CORS(app)

# Load legal PDFs
ipc_pdf_loader = PyPDFLoader("../Backend/assets/ipc.pdf")
constitution_pdf_loader = PyPDFLoader("../Backend/assets/constitution.pdf")

ipc_docs = ipc_pdf_loader.load()
constitution_docs = constitution_pdf_loader.load()

# Create a vector store for document retrieval
llm = OllamaLLM(model="llama3.1")
embedding_model = OllamaEmbeddings(
    model=llm,
)
vector_store = FAISS.from_documents(ipc_docs + constitution_docs, embedding_model)

# Create a QA chain with a more detailed prompt
def create_prompt(retrieved_text, question):
    return f"""You are a legal assistant. Analyze the following legal document excerpts and provide a detailed, human-like response to the user's question.

    Document Excerpts:
    {retrieved_text}

    User's Question:
    {question}

    Your Answer:"""

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided', 'status': False}), 400

    try:
        # Retrieve relevant documents based on the question
        retrieved_docs = vector_store.as_retriever().get_relevant_documents(question)
        retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

        # Create a detailed prompt for the LLM
        prompt = create_prompt(retrieved_text, question)
        
        # Get the response from the LLM
        response = llm(prompt)
        return jsonify({'response': response, 'status': True}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': False}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
