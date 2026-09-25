# Lancer avec : c:\Users\aelsa\Documents\git-practice\parcours_exercices\couche2_casse_repare\exercice_12_assistant_lecture_recherche\travail\venv\Scripts\python.exe -m streamlit run "c:\Users\aelsa\Documents\git-practice\parcours_exercices\couche2_casse_repare\exercice_12_assistant_lecture_recherche\travail\main.py"
"""
services/rag_chain.py — palier 3 (niveau 1) : questions/reponses ancrees dans le PDF

LangChain est utilise pour TOUT ce qui prepare le prompt (recherche par similarite,
formatage) - mais PAS pour l'appel au modele lui-meme, qui passe par notre
ClaudeProvider (MOD-3). Voir le commentaire dans run_rag_chain() pour le detail de
pourquoi on n'utilise pas le style "chaine LangChain" (l'operateur |) ici.
"""
from pathlib import Path
import yaml
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate  # gabarit de prompt avec des "trous" a remplir

from providers import ClaudeProvider  # notre wrapper (MOD-3), pas LangChain directement
from .document_service import recuperer_contenu_document

load_dotenv()  # lit le fichier .env du dossier courant et remplit os.environ avec son contenu

# __file__ = .../travail/services/rag_chain.py
# .parent = .../travail/services/  ->  .parent.parent = .../travail/
# Le dossier prompts/ vit a la racine de travail/, pas dans services/ - d'ou les deux
# ".parent" (piege reel rencontre : un seul ".parent" pointait vers services/prompts/,
# qui n'existe pas).
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

with open(PROMPTS_DIR / "qr-rag.yaml", "r", encoding="utf-8") as f:
    prompt_data = yaml.safe_load(f)
PROMPT_TEMPLATE = prompt_data["rag_prompt"]["v1"]["template"]

# __init__ appele UNE SEULE FOIS ici, au chargement du module - le meme "provider"
# (deja configure avec la cle API) est reutilise a chaque appel de run_rag_chain().
Claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")
provider = ClaudeProvider(api_key=Claude_api_key, model="claude-sonnet-5")


def format_docs(docs) -> str:
    """Recolle les chunks retrouves par le RAG en une seule chaine de texte,
    prete a etre injectee dans le prompt (qui attend du texte simple, pas une
    liste d'objets LangChain)."""
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_chain(query: str, db, contexte_expertise: str = "",historique_contexte: str = "") -> str:
    """
    Repond a une question en s'appuyant sur les passages du PDF les plus proches
    (RAG) ET sur le domaine d'expertise choisi par l'utilisateur (contexte/), si
    defini.

    Args:
        query : la question posee par l'utilisateur.
        db : la base vectorielle Chroma deja construite (cf document_service.py,
            add_to_db) - recue en parametre (MOD-2, injection de dependance),
            pas recreee ici.
        contexte_expertise : le domaine d'expertise ("tu es expert en...", saisi
            dans main.py, lu via services/contexte_service.py) - OPTIONNEL : meme
            principe que l'exercice memoire (tuteur, exercice 5) - on l'utilise s'il
            existe, sinon on continue quand meme plutot que de bloquer l'utilisateur.

    Returns:
        La reponse generee par Claude.
    """
    # Generique : retrouver les 5 passages les plus proches, peu importe le domaine.
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Transforme la chaine YAML en objet prompt LangChain - encore utile ici, c'est
    # juste un gabarit avec des "trous" ({context}, {question}, {contexte_expertise}),
    # rien a voir avec l'appel au modele.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # {context} et {contexte_expertise} sont DEUX choses differentes dans le
    # template (piege reel rencontre : le fichier qr-rag.yaml utilisait {context}
    # pour les deux, donc une seule des deux valeurs pouvait jamais y arriver) :
    # - contexte_documentaire = ce que le RAG a retrouve dans le PDF (change a
    #   chaque question, depend de "query")
    # - contexte_expertise = le domaine choisi par l'utilisateur (fixe pour toute
    #   la session, vient de contexte/ via contexte_service.py)
    contexte_documentaire = format_docs(retriever.invoke(query))

    # .format() remplit directement les "trous" du template avec du texte simple -
    # a ce stade, prompt_final est une chaine de caracteres normale, plus un objet
    # LangChain special.
    prompt_final = prompt_template.format(
        context=contexte_documentaire,
        question=query,
        # "non precise" si l'utilisateur n'a pas (encore) defini de contexte -
        # le prompt reste grammaticalement correct, et la question est quand meme
        # traitee (pas de blocage).
        contexte_expertise=contexte_expertise if contexte_expertise else "non précisé",
        historique_contexte=historique_contexte if historique_contexte else "non précisé",
    )

    # POURQUOI PAS UNE CHAINE LANGCHAIN (l'operateur |) ICI :
    # L'ancienne version faisait :
    #     rag_chain = {"context": ..., "question": ...} | prompt_template | provider | output_parser
    #     response = rag_chain.invoke(query)
    # Ca chaine plusieurs objets avec |, mais CA SUPPOSE que chaque maillon parle le
    # meme "protocole" LangChain (Runnable, avec une methode .invoke()). Notre
    # ClaudeProvider n'est PAS un objet LangChain - c'est notre propre classe (MOD-3),
    # avec sa propre methode .generate(). Le brancher dans un | LangChain ne
    # fonctionne pas (erreur, ou comportement incorrect), un peu comme essayer de
    # visser une prise electrique europeenne dans une prise americaine : meme
    # "famille" d'objet en apparence, protocole different en interne.
    #
    # A la place : on utilise LangChain pour PREPARER le texte (recherche +
    # formatage, ci-dessus), puis on appelle notre provider directement, comme une
    # fonction Python normale - plus simple a suivre, et ca respecte MOD-3 (le reste
    # du code ne connait que ClaudeProvider.generate(), jamais LangChain au-dela de
    # la preparation du prompt).
    return provider.generate(prompt_final)


def analyser_document(db, contexte_expertise: str) -> tuple[bool, str]:
    """
    Palier 2 (niveau 1) + garde-fou "document en lien avec le contexte" - UN SEUL
    appel LLM fait les deux choses (evite un appel redondant) :
    1. Verifie si le document uploade correspond au contexte d'expertise declare.
    2. Si oui, produit une synthese structuree (points cles).

    A appeler juste apres un upload reussi (add_to_db), avant d'autoriser
    l'utilisateur a poser des questions - voir main.py.

    Returns:
        (pertinent, message) :
        - pertinent=True, message=la synthese, si le document correspond au contexte
        - pertinent=False, message=l'explication, sinon (pas de question autorisee)
    """
    with open(PROMPTS_DIR / "synthese.yaml", "r", encoding="utf-8") as f:
        prompt_data_synthese = yaml.safe_load(f)
    template_synthese = prompt_data_synthese["rag_prompt"]["v1"]["template"]

    # recuperer_contenu_document (document_service.py) : logique de recuperation
    # extraite le 2026-09-24, reutilisee aussi par genere_script_2voix
    # (podcast_service.py) - voir le commentaire au-dessus de sa definition pour le
    # detail (db.get() vs recherche par similarite).
    contexte_documentaire = recuperer_contenu_document(db)

    prompt_template = ChatPromptTemplate.from_template(template_synthese)
    prompt_final = prompt_template.format(
        context=contexte_documentaire,
        contexte_expertise=contexte_expertise if contexte_expertise else "non précisé",
    )

    reponse = provider.generate(prompt_final)

    # Le prompt (synthese.yaml) impose "PERTINENT: oui" ou "PERTINENT: non" en
    # toute premiere ligne - on la lit pour decider, sans reinterroger le modele
    # separement avec un deuxieme appel.
    premiere_ligne = reponse.strip().splitlines()[0].lower()
    pertinent = "pertinent: oui" in premiere_ligne

    return pertinent, reponse
