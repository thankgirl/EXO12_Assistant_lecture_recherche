"""
core/context_manager.py — troncature explicite du contexte envoye au LLM (CTX-1)

Particulierement important ici : un document de recherche long peut depasser la
fenetre de contexte du modele. Sans ce module, troncature silencieuse -> reponse
partielle sans avertissement.

A COMPLETER : verification de taille avant l'appel LLM, troncature explicite + log
context_truncated si necessaire.
"""

from pathlib import Path
import yaml
import json 
import os
from dotenv import load_dotenv
from interfaces.schemas import StructureConversation, StructureContext
#from services.rag_chain import run_rag_chain
from providers.LLM_providers import ClaudeProvider  # notre wrapper (MOD-3), pas LangChain directement
from providers.vector_store_provider import get_embedding_model
from langchain_core.prompts import ChatPromptTemplate

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

SETTINGS_DIR = Path(__file__).parent.parent / "config"
with open(SETTINGS_DIR / "settings.yaml", "r", encoding="utf-8") as f:
    settings_param = yaml.safe_load(f)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

with open(PROMPTS_DIR / "resume_contexte.yaml", "r", encoding="utf-8") as f:
    prompt_resume = yaml.safe_load(f)
PROMPT_resume_template = prompt_resume["resume_contexte"]["v1"]["template"]

with open(PROMPTS_DIR / "extract_contexte_pertinent.yaml", "r", encoding="utf-8") as f:
    prompt_extract = yaml.safe_load(f)
PROMPT_extract_template = prompt_extract["extract_contexte_pertinent"]["v1"]["template"]

seuil_resume_tokens= settings_param["limites"]["seuil_resume_tokens"]  # palier 1 : au-dela, resumer les echanges anciens
seuil_recherche_selective_tokens= settings_param["limites"]["seuil_recherche_selective_tokens"] # palier 2 : au-dela, RAG sur l'historique plutot que tout injecter
seuil_nouvelle_conversation_tokens= settings_param["limites"]["seuil_nouvelle_conversation_tokens"] # palier 3 : au-dela, proposer/forcer une nouvelle conversation

Claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")



def resume_contexte (contexte,query) -> str:
    provider = ClaudeProvider(api_key=Claude_api_key, model="claude-sonnet-5")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_resume_template)
        # .format() remplit directement les "trous" du template avec du texte simple -
        # a ce stade, prompt_final est une chaine de caracteres normale, plus un objet
        # LangChain special.
    prompt_final = prompt_template.format(
            contexte=contexte,
            query=query,
        )

    contexte_resume = provider.generate(prompt_final)
    return contexte_resume



def recherche_contexte_pertinent(echanges: list[StructureContext], query: str, top_k: int = 5) -> str:
    # echanges (liste de StructureContext), pas le texte deja assemble : il faut
    # l'embedding de CHAQUE echange individuellement pour pouvoir les classer,
    # perdu si on ne recoit que la chaine concatenee - d'ou le changement d'appel
    # cote context_management (echanges de la conversation, pas contexte_documentaire).
    embeddings = get_embedding_model()  # meme config que ChromeDb, MOD-3 (partagee)

    try:
        # embed_query() renvoie UN vecteur (liste de floats) - cosine_similarity
        # attend des tableaux 2D (une ligne = un element), d'ou le reshape(1, -1)
        # pour transformer un vecteur "plat" en tableau a une seule ligne.
        query_matrix = np.array(embeddings.embed_query(query)).reshape(1, -1)
        # embed_documents() renvoie deja une liste de vecteurs -> tableau 2D direct.
        echanges_matrix = np.array(embeddings.embed_documents([e.texte for e in echanges]))
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings: {e}")

    # similarity_matrix a la forme (1, nb_echanges) : une seule ligne (la query),
    # une colonne par echange - similarity_matrix[0] = le score de chaque echange
    # face a la query.
    similarity_matrix = cosine_similarity(query_matrix, echanges_matrix)
    scores = similarity_matrix[0]

    # argsort trie du plus petit au plus grand score -> [::-1] pour inverser
    # (du plus similaire au moins similaire), puis on garde les top_k premiers
    # indices. Pas de LLM ici (contrairement a resume_contexte) : comme le RAG
    # documentaire (rag_chain.py), selectionner les passages pertinents est de la
    # recherche, pas de la generation - decision prise le 2026-08-29, a confirmer.
    meilleurs_indices = np.argsort(scores)[::-1][:top_k]

    return "\n\n".join(f"{echanges[i].role}: {echanges[i].texte}" for i in meilleurs_indices)
             
    
def context_management(context_file_path:str,query) -> tuple[str,str]:

    provider = ClaudeProvider(api_key=Claude_api_key, model="claude-sonnet-5")
    #uploaded_file.mkdir(parents=True, exist_ok=True)
    conversation_existante=[]
    try:
        with open(context_file_path, "r", encoding="utf-8") as f:
            data = json.load(f) # → un seul dict.
            contexte_histo = StructureConversation(**data) # → un seul objet.

            # Bug trouve le 2026-09-17 : premiere question d'une conversation toute
            # neuve -> echanges=[] -> contexte_documentaire="" -> l'API Anthropic
            # refuse de compter les tokens d'un contenu vide (BadRequestError
            # "messages.0: user messages must have non-empty content"). Rien a
            # resumer/selectionner dans un historique vide - court-circuite avant
            # meme d'appeler count_tokens.
            if not contexte_histo.echanges:
                return "contexte_normal", ""

            # contexte_documentaire = "\n\n".join(context_extract) # context_extract est un objet StructureConversation, pas une liste de chaînes. Il faut parcourir .echanges (la liste des échanges à l'intérieur), pas l'objet entier — quelque chose comme
            contexte_documentaire = "\n\n".join(f"{e.role}: {e.texte}" for e in contexte_histo.echanges)
            nombre_contexte = provider.count_tokens(contexte_documentaire)
            
            if nombre_contexte > seuil_resume_tokens and nombre_contexte <= seuil_recherche_selective_tokens:
                return "contexte_resume", resume_contexte (contexte_documentaire, query) 
            
            elif nombre_contexte > seuil_recherche_selective_tokens and nombre_contexte <= seuil_nouvelle_conversation_tokens:
                    # contexte_histo.echanges (la liste), pas contexte_documentaire (la
                    # chaine deja assemblee) - recherche_contexte_pertinent a besoin de
                    # comparer chaque echange individuellement.
                    return "contexte_pertinent",recherche_contexte_pertinent(contexte_histo.echanges,query)
            
            elif nombre_contexte > seuil_nouvelle_conversation_tokens:
                return "depassement_taille_max","relancer une nouvelle conversation"
            else:
                return "contexte_normal",contexte_documentaire

    except FileNotFoundError:
        print(f"Contexte inconnu avec ce fichier: {context_file_path}")
        return  # rien de valide a mettre a jour - inutile de continuer vers le 2e bloc
        
    
        # try:
        #     with open(uploaded_file, "w", encoding="utf-8") as f:
        #         json.dump(contexte_resume.model_dump(mode="json"), f, ensure_ascii=False, indent=2) #mise à jour avec pydan
        # except Exception as e:
        #     print(f"An error occurred while saving memory: {e}")      

