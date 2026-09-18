"""
services/contexte_service.py — gestion des conversations de l'assistant (MOD-1)

Revu en profondeur le 2026-08-27 (voir architecture.md, section "contexte/ - modele
conversations") : ce n'est plus un texte d'expertise unique par utilisateur, mais une
conversation par document/recherche - une par fichier
LectureAssisteContext_{profil_id}_{id_conversation}.json, id_conversation = UUID court
genere a la creation. Chaque fichier contient UN SEUL objet StructureConversation
(id_conversation, titre, resume, echanges: list[StructureContext]) - pas une liste au
niveau racine, contrairement au pattern tuteur_memoire.py/Souvenir (exercice 5) dont ce
fichier s'inspire par ailleurs. Voir COURS_PYTHON_FICHIERS_LISTES_OBJETS.md (racine du
depot) pour le detail de cette distinction objet-unique/liste et des autres pieges
rencontres en ecrivant ce fichier.

LectureAssisteMemoire.__init__ ne stocke que profil_id (jamais sujet/id_conversation, qui
varient par appel) - lister_conversation() doit pouvoir tourner avant qu'une conversation
soit choisie.

Le sujet/niveau d'expertise "formation" (ex: "expert IA") revient le 2026-08-29, sous
un nom distinct pour ne plus le confondre avec les conversations : "contexte systeme"
(dossier contexte_system/, un seul fichier par profil_id, pas par conversation - suit
l'apprenant de bout en bout, defini une fois). context_system_read(profil_id)/
context_system_write(profil_id, texte) sont des FONCTIONS DE MODULE independantes de
LectureAssisteMemoire (decide le 2026-08-29) - cette classe porte les conversations
ponctuelles de l'utilisateur, le contexte systeme est une donnee de profil a part.

Etat au 2026-08-27 : creer_conversation/lister_conversation/actualiser_conversation
ecrites et debuggees (audits successifs le meme jour).
"""
from pathlib import Path
import glob
from langchain_huggingface import HuggingFaceEmbeddings           # "cerveau n°1" local : texte -> vecteur
from langchain_chroma import Chroma                               # base de données vectorielle locale
import json 
from interfaces.schemas import StructureConversation, StructureContext
import uuid

# __file__ = .../travail/services/contexte_service.py
# .parent = .../travail/services/  ->  .parent.parent = .../travail/
# Meme piege que rag_chain.py : contexte/ vit a la racine de travail/, pas dans
# services/ - il faut bien deux ".parent" pour y arriver depuis ce fichier.
# la convention Python standard (PEP 8) :

# Classes → PascalCase (LectureAssisteMemoire, StructureConversation, ClaudeProvider).
# Fonctions et méthodes → snake_case (creer_conversation, lister_conversation, run_rag_chain).
# Variables → snake_case aussi (context_path, id_conversation).

class LectureAssisteMemoire:

     

    def __init__(self, profil_id: str):
        # initialisation de la classe
        self.profil_id = profil_id
        #self.client = client
        self.context_path = Path(__file__).parent.parent / "LectureAssisteContexte"


    def creer_conversation(self, titre: str, resume: str, echanges: list[StructureContext]) -> None:
        self.id_conversation = uuid.uuid4().hex[:8]
        self.context_sujet_path = self.context_path / f"LectureAssisteContext_{self.profil_id}_{self.id_conversation}.json"

        # UN SEUL objet SourcesLecteurRecherche (pas une liste avec un objet dedans) -
        # c'est lui, une fois serialise, qui devient le contenu entier du fichier.
        # titre/resume/echanges : ceux RECUS en parametre, pas des valeurs figees.
        nouvelle_conversation = StructureConversation(
            id_conversation=self.id_conversation,
            titre=titre,
            resume=resume,
            echanges=echanges,
        )

        # .mkdir() sur context_path lui-meme (le DOSSIER), pas .parent (qui aurait
        # cree travail/ au lieu de LectureAssisteContexte/).
        self.context_path.mkdir(parents=True, exist_ok=True)
        try:
            # context_sujet_path (le FICHIER de cette conversation), pas context_path
            # (le dossier) - ouvrir un dossier en ecriture leve IsADirectoryError.
            with open(self.context_sujet_path, "w", encoding="utf-8") as f:
                json.dump(nouvelle_conversation.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"An error occurred while creating the conversation file: {e}")

    

    #cette fonction va rechercher les conversations déjà enregistrées dans le dossier de l'apprenant
    def lister_conversation(self) -> list[tuple[str, str]]:
       conversations = []
       for fichier in self.context_path.glob(f"LectureAssisteContext_{self.profil_id}_*.json"):
            with open(fichier, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversation = StructureConversation(**data)
                titre = conversation.titre
                id_conversation = conversation.id_conversation
                conversation_tuple = tuple([titre, id_conversation])
                conversations.append(conversation_tuple)

       return conversations
 

    def actualiser_conversation (self, id_conversation: str,profil_id: str, echangesLLM: list[StructureContext]) -> None:

        #  Ajouter un échange plus tard (une future méthode, pas celle-ci) sera différent : lire le SourcesLecteurRecherche
        #  existant depuis son fichier, faire .echanges.append(StructureContext(...)) sur l'objet chargé, puis réécrire l'objet entier
        #  — c'est à ce moment-là que le réflexe "ajouter à une liste puis résauvegarder" de tuteur_memoire.py redevient pertinent,
        #  mais sur .echanges, pas sur le fichier entier.
        self.context_sujet_path = self.context_path / f"LectureAssisteContext_{profil_id}_{id_conversation}.json"
        self.context_path.mkdir(parents=True, exist_ok=True)
        conversation_existante=[]
        try:
            with open(self.context_sujet_path, "r", encoding="utf-8") as f:
                data = json.load(f) # → un seul dict.
                conversation_existante = StructureConversation(**data) # → un seul objet.

            conversation_existante.echanges.extend(echangesLLM)

        except FileNotFoundError:
            print(f"Contexte inconnu avec cet ID conversation: {id_conversation}")
            return  # rien de valide a mettre a jour - inutile de continuer vers le 2e bloc

        try:
            with open(self.context_sujet_path, "w", encoding="utf-8") as f:
                json.dump(conversation_existante.model_dump(mode="json"), f, ensure_ascii=False, indent=2) #mise à jour avec pydan
        except Exception as e:
            print(f"An error occurred while saving memory: {e}")



# contexte SYSTEME (expertise, definie une fois, un seul fichier par profil_id) :
# fonctions independantes de LectureAssisteMemoire (decide le 2026-08-29) - celle-ci
# porte les conversations ponctuelles de l'utilisateur, le contexte systeme est une
# donnee de profil a part, pas couplee aux conversations. profil_id recu en parametre
# explicite (pas de self, pas de classe) a chaque appel.
CONTEXT_SYSTEM_DIR = Path(__file__).parent.parent / "contexte_system"


def context_system_read(profil_id: str) -> str:
    """Lit le contexte systeme (expertise) deja enregistre pour ce profil_id.
    Renvoie une chaine vide si le fichier n'existe pas encore (premier lancement,
    aucun contexte systeme defini)."""
    chemin = CONTEXT_SYSTEM_DIR / f"context_system_{profil_id}.txt"
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def context_system_write(profil_id: str, texte: str) -> None:
    """Enregistre le contexte systeme pour ce profil_id - cree le dossier
    contexte_system/ s'il n'existe pas encore (premier lancement)."""
    CONTEXT_SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    chemin = CONTEXT_SYSTEM_DIR / f"context_system_{profil_id}.txt"
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(texte.strip())

