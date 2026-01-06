import os
import traceback 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace  
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
 
from dotenv import load_dotenv
import streamlit as st  # Ajout de l'import pour Streamlit

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

# Interface Streamlit
st.title("RAG QA sur PDF avec Mistral")

# Upload du PDF
uploaded_file = st.file_uploader("Téléchargez un fichier PDF", type=["pdf"])

if uploaded_file is not None:
    # Sauvegarde temporaire du PDF
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    with st.spinner("Traitement du PDF et création de la base vectorielle..."):
        chunks = process_pdf(pdf_path)
        if chunks:
            create_vector_db(chunks)
            st.session_state.chain = load_rag_chain()
            st.success("PDF traité et chaîne RAG chargée avec succès !")
        else:
            st.error("Erreur lors du traitement du PDF.")

# Si la chaîne est chargée (dans session_state)
if "chain" in st.session_state:
    question = st.text_input("Posez votre question sur le document :")
    
    if st.button("Répondre"):
        if question:
            with st.spinner("Génération de la réponse en cours..."):
                try:
                    result = st.session_state.chain.invoke({"query": question})
                    st.subheader("Réponse de Mistral :")
                    st.write(result['result'])
                    
                    st.subheader("Sources utilisées :")
                    for doc in result['source_documents']:
                        page = doc.metadata.get('page', '?')
                        content_snippet = doc.page_content[:100] + "..."
                        st.write(f"- Page {page} : {content_snippet}")
                except Exception as e:
                    st.error(f"Erreur lors de la génération : {str(e)}")
        else:
            st.warning("Veuillez entrer une question.")
else:
    st.info("Téléchargez un PDF pour commencer.")

# Optionnel : Nettoyage du fichier temporaire après usage (mais pas obligatoire pour un app simple)
# os.remove(pdf_path) if os.path.exists(pdf_path) else None