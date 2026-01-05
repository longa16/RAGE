import os
import traceback 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace  
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
 
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Chargement et chunking du PDF
def process_pdf(pdf_path):
    print(f"Chargement du fichier: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"Le fichier {pdf_path} n'existe pas.")
        return []
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_documents(documents)
    print("Chunking du document terminé.")
    return chunks

# embedding et création de la base de données vectorielle
def create_vector_db(chunks):
    print("Création de la base de données vectorielle...")
    print("Initialisation du modèle d'embedding...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Vectorisation des chunks en cours...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("Base de données vectorielle créée et sauvegardée localement.")
    return db

def load_rag_chain():
    print("--- 3. Chargement du Cerveau (Mistral 7B) ---")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1, 
    )

    chat_llm = ChatHuggingFace(llm=llm)
    
    # Template de prompt adapté pour le format chat/instruct
    prompt_template = """<s>[INST] Utilise le contexte suivant pour répondre à la question. Si tu ne sais pas, dis-le simplement.

Contexte : {context}

Question : {question} [/INST]</s>"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff", 
        retriever=db.as_retriever(search_kwargs={"k": 3}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


if __name__ == "__main__":
    
    if os.path.exists("faiss_index"):
        chain = load_rag_chain()
        
        question = "Quels sont les points clés de ce document ?"
        print(f"\nQuestion : {question}")
        print("Réflexion en cours...")
        
        resultat = chain.invoke({"query": question})
        
        print("\n--- RÉPONSE DE MISTRAL ---")
        print(resultat['result'])
        
        print("\n--- SOURCES UTILISÉES ---")
        for doc in resultat['source_documents']:
            print(f"- Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")
    else:
        print("Erreur : Lance d'abord la création de la base (code précédent) ou vérifie le dossier faiss_index.")
