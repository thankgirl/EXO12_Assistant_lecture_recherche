"""
interfaces/schemas.py — contrats de donnees valides automatiquement (MOD-4)

A COMPLETER : modeles Pydantic (ex: DocumentCharge, SyntheseStructuree,
ReponseQuestion...) - pas de dictionnaires libres entre rag_chain.py, main.py et
providers/.
"""
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

#Spécification du contexte à sauvegarder
class StructureContext(BaseModel):
    sujet: str
    role: Literal["user", "assistant"]
    texte: str
    date: datetime

#Spécification de la structure d'une conversation
class StructureConversation(BaseModel):
    id_conversation: str
    titre: str
    resume: str
    echanges: list[StructureContext]

#SyntheseStructuree = la sortie structurée de analyser_document() (résumé, points clés, pertinence...)
class StructureSynthese(BaseModel):
    sujet:str
    resume: str
    definitions : str
    objectifs_du_doc : str
    concept_cles : str
    donnees_factuelles : str
    analyse_document : str
    recommandations : str

 
#DocumentCharge = un contrat structuré pour le résultat de validation/upload (file_check/add_to_db), au lieu du tuple (statut, message) actuel. Pas fait, optionnel.
#ReponseQuestion = un contrat structuré pour la réponse de run_rag_chain() (au lieu d'une simple chaîne), par exemple avec les passages sources cités. Pas fait, optionnel.