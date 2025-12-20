import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings    
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from  langchain.chains import RetrievalQA 

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

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
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

    repo_id = "mistral/mistral-7B-instruct-v0.3"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.1,
        max_length=512,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever= db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    return qa_chain

if __name__ == "__main__":

    if os.path.exists("faiss_index"):
        chain = load_rag_chain()

        question = "Quels sont les points clés de ce document ?"
        print(f"Question : {question}")
        print("Génération de la réponse...")

        resultat = chain.invoke({"query": question})

        print("Réponse générée :")
        print(resultat['result'])

        print("\n Portions du pdf utilisées :")
        for doc in resultat['source_documents']:
            print(f"- Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")

else:
    print("Erreur : Lance d'abord la création de la base")
