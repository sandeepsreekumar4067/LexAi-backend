from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import time

print("execution started")
start = time.time()
llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
)
embedding_model = OllamaEmbeddings(model="llama3.1")
# sample chat context
title_model = ChatOllama(
    model="llama3.1",
    temperature=0.1
)
lawyer_model = ChatOllama(
    model="llama3.1",
    temperature=0.1
)

nlp = spacy.load("en_core_web_sm")


chat_context = [
    ("human", "Hi there!"),
    ("ai", "Hello! How can I assist you today?"),
    ("human", "Can you tell me about the weather today?"),
    ("ai", "Sure! The weather today is sunny with a high of 25°C and a low of 15°C."),
    ("human", "Can you tell me about Section 420 of the Indian Penal Code?"),
    (
        "ai",
        "Section 420 of the Indian Penal Code deals with cheating and dishonestly inducing delivery of property. It prescribes imprisonment of up to 7 years and a fine.",
    ),
    ("human", "How do I file a complaint under this section?"),
    (
        "ai",
        "To file a complaint under Section 420, you need to approach the police station and file an FIR providing evidence of the cheating incident.",
    ),
    ("human", "What evidence is required for filing a complaint under Section 420?"),
    (
        "ai",
        "Evidence required includes documents showing financial transactions, witnesses to the cheating incident, and any contracts or written agreements that were breached.",
    ),
    ("human", "How long does it take to resolve such cases?"),
    (
        "ai",
        "The time to resolve a case under Section 420 depends on the complexity, availability of evidence, and court workload. It can take several months to years.",
    ),
    ("human", "What is anticipatory bail, and how does it work?"),
    (
        "ai",
        "Anticipatory bail is a legal provision under Section 438 of the CrPC that allows an individual to seek bail in anticipation of an arrest. It protects the person from being taken into custody.",
    ),
    ("human", "Can a person get anticipatory bail in a Section 420 case?"),
    (
        "ai",
        "Yes, a person can apply for anticipatory bail in a Section 420 case. The court considers factors like the seriousness of the offense and the evidence before granting it.",
    ),
    ("human", "What if the anticipatory bail is rejected?"),
    (
        "ai",
        "If anticipatory bail is rejected, the person can approach a higher court, such as the High Court or the Supreme Court, to appeal the decision.",
    ),
    ("human", "What is the role of a legal notice in case management?"),
    (
        "ai",
        "A legal notice is a formal communication sent to the opposing party to notify them of a legal grievance. It serves as a final warning before initiating legal proceedings.",
    ),
    ("human", "How do I draft a legal notice?"),
    (
        "ai",
        "A legal notice should include the sender's details, recipient's details, a clear description of the grievance, supporting evidence, and a demand for resolution within a specified timeframe.",
    ),
    ("human", "What happens if the recipient ignores the legal notice?"),
    (
        "ai",
        "If the recipient ignores the legal notice, the sender can proceed with filing a case in court. Ignoring the notice may weaken the recipient's defense.",
    ),
    ("human", "Can you explain the process of filing a civil suit?"),
    (
        "ai",
        "To file a civil suit, you need to draft a plaint, pay the court fees, and file it in the appropriate court. The court then issues a summons to the defendant.",
    ),
    ("human", "What is a summons?"),
    (
        "ai",
        "A summons is a legal document issued by the court to notify the defendant of the lawsuit and require their presence in court.",
    ),
    (
        "human",
        "What if the defendant doesn’t appear in court after receiving a summons?",
    ),
    (
        "ai",
        "If the defendant fails to appear, the court may proceed ex parte, meaning it decides the case in the absence of the defendant.",
    ),
    ("human", "Can a civil case be settled out of court?"),
    (
        "ai",
        "Yes, civil cases can often be settled out of court through mediation, arbitration, or mutual agreement between the parties.",
    ),
    ("human", "What is the difference between mediation and arbitration?"),
    (
        "ai",
        "Mediation is a voluntary process where a neutral mediator helps parties reach an agreement. Arbitration is more formal, where an arbitrator makes a binding decision.",
    ),
    ("human", "Can mediation be used in criminal cases?"),
    (
        "ai",
        "Mediation is generally not used in serious criminal cases, but it can be applied in minor disputes, like property disputes or family matters, with the court's approval.",
    ),
    ("human", "Tell me about Public Interest Litigation (PIL)."),
    (
        "ai",
        "Public Interest Litigation (PIL) is a legal action initiated in court for the enforcement of public interest, especially in cases of rights violations or social issues.",
    ),
    ("human", "Who can file a PIL in India?"),
    (
        "ai",
        "Any citizen or organization can file a PIL in the High Court or Supreme Court to address issues affecting the public at large.",
    ),
    ("human", "What are the limitations of filing a PIL?"),
    (
        "ai",
        "Limitations include the need for a genuine public interest, avoidance of frivolous claims, and adherence to the court’s jurisdiction.",
    ),
    (
        "human",
        "Can you explain the importance of the landmark Kesavananda Bharati case?",
    ),
    (
        "ai",
        "The Kesavananda Bharati case established the doctrine of the basic structure of the Constitution, ensuring that Parliament cannot alter its fundamental features.",
    ),
    ("human", "How does this doctrine affect constitutional amendments?"),
    (
        "ai",
        "It restricts Parliament from amending the Constitution in a way that alters its basic structure, such as democracy, secularism, or the rule of law.",
    ),
    ("human", "Can you give an example of a case where this doctrine was applied?"),
    (
        "ai",
        "The Minerva Mills case applied the basic structure doctrine to strike down amendments that violated the principle of judicial review.",
    ),
    ("human", "What is the role of the Supreme Court in ensuring justice?"),
    (
        "ai",
        "The Supreme Court ensures justice by interpreting the Constitution, reviewing laws, and safeguarding fundamental rights. It also acts as the final appellate authority.",
    ),
    ("human", "Can a person directly approach the Supreme Court?"),
    (
        "ai",
        "Yes, a person can directly approach the Supreme Court under Article 32 for enforcement of fundamental rights or through special leave petitions under Article 136.",
    ),
    ("human", "How are judges appointed to the Supreme Court?"),
    (
        "ai",
        "Supreme Court judges are appointed by the President of India based on the recommendations of the Collegium, which includes senior judges and the Chief Justice of India.",
    ),
    ("human", "What is judicial activism?"),
    (
        "ai",
        "Judicial activism refers to the proactive role of the judiciary in protecting rights and addressing issues where the executive or legislature fails to act.",
    ),
    ("human", "Can judicial activism be criticized?"),
    (
        "ai",
        "Yes, critics argue that judicial activism can lead to judicial overreach, where courts encroach upon the roles of the executive or legislature.",
    ),
    (
        "human",
        "Can you summarize what we discussed about case management and legal entities?",
    ),
    (
        "ai",
        "Certainly! We covered the roles of legal notices, filing processes for civil and criminal cases, anticipatory bail, mediation vs. arbitration, Public Interest Litigation, the Kesavananda Bharati case, and judicial activism. These topics highlight essential aspects of law enforcement and case management in India.",
    ),
    ("human", "Thanks! This was very helpful."),
    (
        "ai",
        "You’re welcome! Feel free to reach out if you have more questions in the future. Have a great day!",
    ),
]


summarisation_template = PromptTemplate.from_template(
    """
        Summarize the following chat conversation. Focus on the key points, user intentions, and AI responses:
        dont make it too long . make it within 120 words.
        Chat History : {chat_history}
        Summary:
    """
)

title_template = PromptTemplate.from_template(
    """
    You are assigned with a task to find the most appropriate title to the summary that will be providing.
    make it within 2 to 3 words. most of the summary will be related to the law 
    Summary : {summary}
    """
)

proficiency_field_template = PromptTemplate.from_template(
    """
    i will provide you a summary .
    based on the summary find the most appropriate type of lawyer that can be assigned to this case.
    strictly select the lawyer from these 5 types : ["Civil Lawyer","Criminal Lawyer","Corporate Lawyer","Family Lawyer","Intellectual Property (IP) Lawyer"].
    do not select one other than these 5 types .
    these are the description for each lawyer type: 
    Civil Lawyer : Handles non-criminal disputes like property, contracts, and consumer rights cases.
    Criminal Lawyer : Defends or prosecutes individuals accused of crimes like theft, fraud, or assault.
    Corporate Lawyer : Advises businesses on legal compliance, contracts, mergers, and corporate governance.
    Family Lawyer : Deals with divorce, child custody, adoption, and domestic disputes.
    Intellectual Property (IP) Lawyer : Protects patents, trademarks, copyrights, and trade secrets.
    Summary : {summary}
    """
)

tokenisation_template = PromptTemplate.from_template(
    """
        Tokenize the following chat conversation . focus on the key points , user intentions and AI response , 
        Extract IMPORTANT tokens from the given conversation : {chat_history},
        Tokens:
    """
)


def summariseChat(chat_history):
    chat_as_text = "\n".join([f"{role} : {message}" for role, message in chat_history])
    summarised_prompt = summarisation_template.format(chat_history=chat_as_text)
    return llm.invoke(summarised_prompt).content
