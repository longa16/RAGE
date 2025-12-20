import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings    
from langchain_community.vectorstores import FAISS


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

if __name__ == "__main__":

    mon_fichier= "data/AI.pdf"

    chunks_result = process_pdf(mon_fichier)

    if chunks_result:
        create_vector_db(chunks_result)
        print("\n--- Test de recherche rapide ---")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        query = "Quel est le theme de ce document?"
        docs = new_db.similarity_search(query, k=2)

        print(f"Questions: {query}")
        print(f"Réponses trouvées:{docs[0].page_content} ")
        