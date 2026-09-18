"""
providers/Vector_providers.py — fournisseurs de base vectorielles isolées derriere une interface (MOD-3)

La logique metier (services/, updload_file.py a venir) ne connaitra que le contrat
de la base (init + generate) - jamais le fournisseur directement. Changer de fournisseur =
ajouter une classe ici, zero ligne a changer dans services/.
"""
from abc import ABC, abstractmethod  # module standard Python pour definir un "contrat"
import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings           # "cerveau n°1" local : texte -> vecteur
from langchain_chroma import Chroma                               # base de données vectorielle locale
from langchain_community.document_loaders import TextLoader       #loarder de fichiers TXT
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter  # découpe le texte en chunks
import hashlib  # CTX-2 : IDs déterministes pour rendre l'ajout en base idempotent

def get_embedding_model(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> HuggingFaceEmbeddings:
    """Construit le modele d'embeddings - fonction simple (pas de classe : rien a
    conserver entre deux appels au-dela de l'objet lui-meme), partagee entre ChromeDb
    (recherche documentaire, PDF) et context_manager.py (recherche dans l'historique
    d'une conversation) - MOD-3, un seul endroit pour ce choix. Tourne en local, pas
    besoin de cle API. model_name aligne par defaut sur celui deja utilise pour le
    decoupage en chunks (document_service.py, SentenceTransformersTokenTextSplitter).
    """
    return HuggingFaceEmbeddings(model_name=model_name)


class VectorProvider(ABC):
    """
    Le CONTRAT commun a tous les fournisseurs de base vectorielles - pas une classe qu'on utilise
    directement, juste la liste des methodes que chaque fournisseur DOIT avoir.
    'ABC' = Abstract Base Class : Python empeche de creer un objet VectorProvider() tout
    seul (ça n'aurait pas de sens, generate() n'a pas de corps ici) - on doit passer
    par une sous-classe concrete comme ClaudeProvider, qui elle fournit le vrai code.

    @abstractmethod = oblige chaque sous-classe a definir generate(), sinon Python
    refuse de la laisser s'instancier (erreur au demarrage, pas en pleine utilisation).
    """

    @abstractmethod
    def upload_userfile(self, file_dir: str, **kwargs) -> Chroma:
        # Renomme le 2026-08-30 (etait "generate", copie-colle du contrat LLMProvider
        # qui ne correspond a rien de ce qu'un fournisseur de base vectorielle fait
        # reellement) - ChromeDb.upload_userfile() est la seule methode concrete
        # existante, c'est elle qui devient le contrat. **kwargs : meme idee que
        # LLMProvider - capture des parametres futurs sans casser la signature.
        ...


class ChromeDb(VectorProvider):
    """Fournisseur concret pour la base db chrome"""

    def __init__(self, collection_name: str , persist_directory: str = "./download_db"):
        # collection_name/persist_directory parametres (pas en dur) : permet a main.py
        # (MOD-2, l'assemblage) de choisir le nom - identifiant simple pour l'instant
        # (decide le 2026-08-27, pas le coeur de cet exercice) ; un vrai identifiant
        # par utilisateur/session viendra avec l'exercice multi-utilisateur a venir.
        # __init__ est appele UNE SEULE FOIS, au moment ou on cree l'objet :
        #     provider = ClaudeProvider(api_key="...", model="claude-sonnet-5")
        # Il configure l'objet (ici : prepare le client ChatAnthropic) et le stocke
        # dans self._model pour que generate() puisse s'en servir plus tard.
        # api_key recu en PARAMETRE (pas lu ici via os.getenv) : c'est le bon reflexe
        # MOD-2 - le composant reçoit sa dependance (la cle), il ne va pas la
        # chercher lui-meme dans l'environnement. C'est main.py (l'assemblage) qui
        # lira os.getenv("ANTHROPIC_API_KEY") et le passera ici.
        embedding_model = get_embedding_model()
        # self.db (pas juste "db") : sans le "self.", la base créée ici disparaît dès
        # que __init__ se termine - une variable locale ne survit pas à sa méthode.
        # upload_userfile() a besoin de retrouver la MEME instance plus tard.
        self.db = Chroma(collection_name=collection_name,  # nom interne de la collection , en ajoutant self., la variable est globale
                         #variable locale à __init__, jamais stockée sur self, donc perdue dès que __init__ se termine
        embedding_function=embedding_model,  # modèle utilisé pour transformer texte -> vecteur
        persist_directory=persist_directory)


    def upload_userfile(self,  file_dir: str, **kwargs) -> Chroma: # annotation de type, pas juste un rappel de nom, défini le type en sorti
        # generate() est appele CHAQUE FOIS qu'on veut une reponse, potentiellement
        # plusieurs fois avec le meme objet provider deja configure :
        #     reponse1 = provider.generate("premiere question")
        #     reponse2 = provider.generate("deuxieme question")
        # .invoke(prompt) = methode LangChain qui envoie le prompt et attend la
        # reponse. .content = extrait le texte brut de l'objet reponse renvoye.
        for file_path in glob.glob(os.path.join(file_dir, "*.txt")):
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                # chargement des fichiers TXT
            loader = TextLoader(file_path)
            data = loader.load()   

            # Store metadata and content
            doc_metadata = [data[i].metadata for i in range(len(data))]
            doc_content = [data[i].page_content for i in range(len(data))]

            # Générique : découpage en chunks d'environ 100 mots avec chevauchement de 50,
            st_text_splitter = SentenceTransformersTokenTextSplitter(
                model_name="sentence-transformers/all-mpnet-base-v2",
                chunk_size=100,
                chunk_overlap=50
            )
            st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

            # Chaque chunk est vectorisé (via embedding_model) et stocké dans Chroma.
            # CTX-2 (idempotence) : @st.cache_resource évite de recharger pendant qu'un même
            # processus tourne, mais rome_db persiste sur disque - à chaque redémarrage du
            # serveur, ce bloc s'exécute à nouveau. Sans ID déterministe, chaque redémarrage
            # dupliquerait les fiches ROME dans la base déjà existante.
            nom_fichier = os.path.basename(file_path)
            chunk_ids = [
                hashlib.sha256(f"{nom_fichier}-{i}-{chunk.page_content}".encode()).hexdigest()
                for i, chunk in enumerate(st_chunks)
            ]
            self.db.add_documents(st_chunks, ids=chunk_ids)

        return self.db #c'est le même self.db en sortie et non juste db qui est une variable non reconnue
        



