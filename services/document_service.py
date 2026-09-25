"""
services/document_service.py — traite un fichier uploade : valide, puis extrait (MOD-1)

"""
import os
from pathlib import Path
import hashlib
from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader,UnstructuredExcelLoader,Docx2txtLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from core.file_validator import file_check

def format_docs(docs):
    """Formats a list of document objects into a single string.

    Args:
        docs (list): A list of document objects, each having a 'page_content' attribute.

    Returns:
        str: A single string containing the page content from each document,
        separated by double newlines."""
    # Recolle tous les chunks trouvés en une seule chaîne de texte, prête à être
    # injectée dans un prompt (qui attend du texte simple, pas une liste d'objets).
    return "\n\n".join(doc.page_content for doc in docs)


def add_to_db(uploaded_files, db, dossier_documents=None) -> tuple[str, str | None]:
    """Traite les fichiers uploades et les ajoute a la base vectorielle.

    Args:
        uploaded_files : fichiers recus du st.file_uploader (main.py).
        db : la base vectorielle Chroma (MOD-2, injection - creee dans main.py,
            pas ici, pour que run_rag_chain/analyser_document utilisent la MEME
            instance ensuite).
        dossier_documents : dossier ou deplacer le fichier source une fois indexe
            (2026-09-17), au lieu de le supprimer - permet de retrouver le document
            d'origine des semaines plus tard (une conversation sans son document
            source n'est pas exploitable, seul le contenu decoupe/vectorise
            survivait jusqu'ici). None (defaut) garde l'ancien comportement
            (suppression) - main.py doit fournir un dossier propre a la
            conversation active pour beneficier du stockage.

    Returns:
        (statut, resultat) : ("success", None) si tout s'est bien passe,
        ("error", message) sinon - main.py affiche resultat via st.error(...) en
        cas d'echec (une fonction de services/ ne doit pas appeler st.error()
        elle-meme, MOD-1 : ce n'est pas sa responsabilite, juste retourner
        l'information).
    """
    if not uploaded_files:
        return "error", "Aucun fichier fourni."
    
    for uploaded_file in uploaded_files:
        status, message = file_check(uploaded_file)
        if status!="success" :
            return status, message
        
        # Streamlit reçoit le fichier uploadé en mémoire, pas comme un vrai fichier sur
        # disque - on doit d'abord l'écrire quelque part pour que PyPDFLoader (qui attend
        # un chemin de fichier) puisse le lire.
        #temp_file_path = os.path.join("./temp", uploaded_file.name)
        # temp_file_path = Path(__file__).parent.parent/"temp"/{uploaded_file}.name
        # ^ {uploaded_file} sans prefixe f est un ENSEMBLE (set) Python, pas une
        # interpolation - .name dessus plante (AttributeError). Path se construit en
        # enchainant des "/" avec chaque morceau separe, pas avec une chaine a
        # formater :
        temp_file_path = Path(__file__).parent.parent / "temp" / uploaded_file.name
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)  # crée le dossier ./temp s'il n'existe pas déjà

        with open(temp_file_path, "wb") as temp_file:   # "wb" = écriture en mode binaire
            temp_file.write(uploaded_file.getbuffer())  # récupère les octets bruts du fichier uploadé et les écrit

        if temp_file_path.suffix.lower()==".pdf" :
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()   # renvoie une liste d'objets "page" (un par page du PDF)

            # metadata = infos annexes (ex: numéro de page, nom du fichier source) associées à chaque page ;
            # content = le texte brut extrait de chaque page.
            doc_metadata = [data[i].metadata for i in range(len(data))]
            doc_content = [data[i].page_content for i in range(len(data))]

        elif temp_file_path.suffix.lower() ==".doc":
            loader = Docx2txtLoader(temp_file_path)
            data = loader.load()   # renvoie une liste d'objets "page" (un par page du PDF)

            # metadata = infos annexes (ex: numéro de page, nom du fichier source) associées à chaque page ;
            # content = le texte brut extrait de chaque page.
            doc_metadata = [data[i].metadata for i in range(len(data))]
            doc_content = [data[i].page_content for i in range(len(data))]

        elif temp_file_path.suffix.lower() ==".txt":
            loader = TextLoader(temp_file_path)
            data = loader.load()   # renvoie une liste d'objets "page" (un par page du PDF)

            # metadata = infos annexes (ex: numéro de page, nom du fichier source) associées à chaque page ;
            # content = le texte brut extrait de chaque page.
            doc_metadata = [data[i].metadata for i in range(len(data))]
            doc_content = [data[i].page_content for i in range(len(data))]

        elif temp_file_path.suffix.lower() ==".csv":
            loader = CSVLoader(temp_file_path)
            data = loader.load()   # renvoie une liste d'objets "page" (un par page du PDF)

            # metadata = infos annexes (ex: numéro de page, nom du fichier source) associées à chaque page ;
            # content = le texte brut extrait de chaque page.
            doc_metadata = [data[i].metadata for i in range(len(data))]
            doc_content = [data[i].page_content for i in range(len(data))]

        elif temp_file_path.suffix.lower() ==".xls":
            loader = UnstructuredExcelLoader(temp_file_path, mode="elements")  
                # mode="elements" keeps cell-level granularity; mode="single" merges into one doc
            data = loader.load()

            # metadata = infos annexes (ex: numéro de page, nom du fichier source) associées à chaque page ;
            # content = le texte brut extrait de chaque page.
            doc_metadata = [data[i].metadata for i in range(len(data))]
            doc_content = [data[i].page_content for i in range(len(data))]
        # Découpage en petits morceaux ("chunks") d'environ 100 mots (tokens), avec un
        # chevauchement de 50 mots - le chevauchement évite de couper une idée exactement
        # en deux entre la fin d'un chunk et le début du suivant.
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",  # modèle utilisé pour compter les tokens correctement
            chunk_size=100,
            chunk_overlap=50,
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

        # Chaque chunk est transformé en vecteur et stocké dans Chroma, prêt à être
        # retrouvé plus tard par une recherche de similarité.
        # CTX-2 (idempotence) : ID déterministe - re-uploader deux fois le même
        # document ne doit pas le dupliquer dans la base.
        chunk_ids = [
            hashlib.sha256(f"{uploaded_file.name}-{i}-{chunk.page_content}".encode()).hexdigest()
            for i, chunk in enumerate(st_chunks)
        ]
        db.add_documents(st_chunks, ids=chunk_ids)

        # Stockage persistant du fichier source (2026-09-17) : deplace vers
        # dossier_documents plutot que supprime, si un dossier a ete fourni - sinon
        # (dossier_documents=None) on garde l'ancien comportement (suppression).
        # .replace() deplace/renomme le fichier (pas une copie) - le temporaire
        # n'existe plus a cet emplacement une fois deplace.
        if dossier_documents is not None:
            dossier_documents.mkdir(parents=True, exist_ok=True)
            temp_file_path.replace(dossier_documents / uploaded_file.name)
        else:
            os.remove(temp_file_path)

    return "success", None
   
# 🎓 À RETENIR : db.get() vs db.as_retriever().invoke(query) (voir run_rag_chain,
# rag_chain.py) - deux facons differentes d'interroger Chroma. .get() recupere TOUT
# le contenu stocke, sans notion de similarite ni de question - utile quand on veut
# une vue d'ensemble du document entier (synthese, script de podcast).
# .as_retriever().invoke(query) fait une recherche par similarite semantique, et ne
# renvoie que les k passages les plus proches d'une question precise - utile pour
# repondre a une question ciblee, pas pour tout relire.
# Extraite ici le 2026-09-24 (etait avant dupliquee dans analyser_document) pour que
# genere_script_2voix (podcast_service.py) puisse la reutiliser sans recopier la
# logique - le cout de rappeler cette fonction deux fois est nul (lecture locale de
# Chroma, pas un appel LLM), donc pas besoin de faire circuler le resultat entre les
# deux boutons/reruns Streamlit.
def recuperer_contenu_document(db) -> str:
    """Recupere l'integralite du contenu stocke pour cette conversation.

    Args:
        db : la base vectorielle Chroma (meme instance que celle utilisee par
            add_to_db/run_rag_chain/analyser_document, injectee depuis main.py).

    Returns:
        Tous les chunks du document, recolles en une seule chaine de texte.
    """
    tout_le_contenu = db.get()["documents"]
    return "\n\n".join(tout_le_contenu)
