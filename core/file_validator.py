"""
core/file_validator.py — validation type/taille des fichiers entrants (CTX-3)

Uniquement la validation - rejet immediat si le fichier ne passe pas, avant de
consommer des ressources ou des tokens LLM. L'extraction du contenu (une fois le
fichier valide) est une responsabilite differente -> services/document_service.py
(MOD-1 : une unite, une responsabilite).

Regles (types autorises, taille max) a lire depuis
governance/guardrails/input_guardrails.yaml - jamais codees en dur ici.

A COMPLETER : fonction valider_fichier(fichier) -> bool, appelee par
services/document_service.py (services/ peut appeler core/, sens autorise par MOD-6)
avant toute extraction. main.py ne connait pas cette etape, juste le resultat final.
"""

from pathlib import Path
import yaml
from dotenv import load_dotenv
import os
from pypdf import PdfReader

GUARDRAILS_DIR = Path(__file__).parent.parent / "governance/guardrails/"
with open(GUARDRAILS_DIR / "input_guardrails.yaml", "r", encoding="utf-8") as f:
    guardrails_param = yaml.safe_load(f)

SETTINGS_DIR = Path(__file__).parent.parent / "config"
with open(SETTINGS_DIR / "settings.yaml", "r", encoding="utf-8") as f:
    settings_param = yaml.safe_load(f)

max_pages_pdf= settings_param["limites"]["max_pages_pdf"]

taille_max_mo = guardrails_param["fichiers_document"]["taille_max_mo"]

def file_check(uploaded_file) -> tuple[str,str]:

    if uploaded_file.size > taille_max_mo* 1024 * 1024:
        return "error", f"{uploaded_file.name} dépasse {taille_max_mo} Mo — traitement arrêté."

    if Path(uploaded_file.name).suffix.lower() not in guardrails_param["fichiers_document"]["types_autorises"]:
        return "error", f"{uploaded_file.name} n'a pas le type attendu : pdf, doc, txt, xls"

    if Path(uploaded_file.name).suffix.lower() == ".pdf" :
        # PyPDFLoader exige un vrai chemin sur disque - uploaded_file est encore en
        # memoire a ce stade (rien ecrit sur disque avant document_service.py).
        # PdfReader (pypdf) sait lire un objet fichier en memoire directement.
        pdf_nb_pages = len(PdfReader(uploaded_file).pages)
        if pdf_nb_pages > max_pages_pdf :
            return "error", f"Le fichier pdf {uploaded_file.name} contient {pdf_nb_pages} pages, dépasse la limite autorisée {max_pages_pdf}"
    
    return "success","Tout est OK"
