"""
services/podcast_service.py — niveau 2, palier 4 (script) + 6 (podcast audio) :

Genere un podcast audio a partir du contenu d'un document deja uploade/analyse
(niveau 1). Deux etapes distinctes (voir ENONCE.md, section Niveau 2) :
- palier 4 : script de dialogue pedagogique a deux voix (texte -> texte, ici)
- palier 6 : synthese vocale du script (texte -> audio, a construire ensuite,
  Dia2 - voir travail/audio_local/podcast_deux_voix.py pour l'infra deja testee)

# 🎓 À RETENIR : pourquoi un service separe de rag_chain.py plutot qu'une fonction
# de plus dans analyser_document/run_rag_chain - generer un podcast est une action
# a la demande de l'utilisateur (bouton dedie, pas automatique a chaque analyse),
# avec son propre cout token (le prompt developpe les concepts, contrairement au
# resume compact de synthese.yaml) - la separer en son propre fichier/service rend
# ce choix explicite dans l'architecture, pas juste dans le comportement de l'UI.
"""
from pathlib import Path
import yaml
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate  # gabarit de prompt avec des "trous" a remplir
from providers import ClaudeProvider  # notre wrapper (MOD-3), pas LangChain directement
from providers import get_tts_provider
from .document_service import recuperer_contenu_document
import json 
from interfaces.schemas import StructureConversation

load_dotenv()  # lit le fichier .env du dossier courant et remplit os.environ avec son contenu

# __file__ = .../travail/services/podcast_service.py
# .parent = .../travail/services/  ->  .parent.parent = .../travail/
# Le dossier prompts/ vit a la racine de travail/, pas dans services/ - meme piege
# que rag_chain.py, voir son commentaire pour le detail.
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# __init__ appele UNE SEULE FOIS ici, au chargement du module - meme principe que
# rag_chain.py (chaque service garde sa propre instance pour l'instant ; les deux
# partagent la meme cle API/le meme modele, une factorisation possible plus tard
# si un 3e service apparait, pas urgent a 2).
Claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")
provider = ClaudeProvider(api_key=Claude_api_key, model="claude-sonnet-5")




def genere_script_2voix(db, dossier_documents: Path, document_name: str, niveau_public="", ton="", points_particuliers="") -> tuple[str, str]:
    """Genere un script de dialogue pedagogique a deux voix a partir du document.

    Args:
        db : la base vectorielle Chroma de la conversation active (meme instance
            que add_to_db/run_rag_chain/analyser_document).
        dossier_documents : dossier ou ecrire le script (ex: le sous-dossier
            voice/ de documents_{id_conversation}/, cree par l'appelant si besoin).
        document_name : nom du fichier a ecrire (ex: "script_dialogue.txt").

    Returns:
        "succes" ou "echec" - voir le TODO plus bas, pas encore aligne sur le
        pattern (statut, message) du reste du projet (add_to_db, analyser_document).
    """
    with open(PROMPTS_DIR / "script_dialogue.yaml", "r", encoding="utf-8") as f:
        prompt_data_script = yaml.safe_load(f)
    template_script = prompt_data_script["script_prompt"]["v1"]["template"]

    # recuperer_contenu_document (document_service.py) : TOUT le contenu du
    # document, pas une recherche par similarite - le script doit pouvoir
    # developper n'importe quel concept du document, pas seulement ceux proches
    # d'une question precise (voir le commentaire au-dessus de sa definition).
    contexte_documentaire = recuperer_contenu_document(db)

    prompt_template = ChatPromptTemplate.from_template(template_script)
    prompt_final = prompt_template.format(
        contexte_documentaire=contexte_documentaire,
        niveau_public=niveau_public if niveau_public else "non précisé",
        ton=ton if ton else "non précisé",
        points_particuliers=points_particuliers if points_particuliers else "non précisé"
    )

    reponse = provider.generate(prompt_final)

    # TODO (releve en audit le 2026-09-24, pas corrige - a decider) :
    # 1. `except:` nu attrape TOUT (bugs de programmation inclus), sans jamais dire
    #    pourquoi - le reste du projet retourne toujours (statut, message) avec le
    #    detail de l'erreur (voir add_to_db). Ici, un "echec" ne dit rien de plus.
    # 2. Cette fonction retourne un statut, pas le script lui-meme - main.py devra
    #    relire le fichier pour l'afficher (ex: dans l'expander de la synthese).
    #    Delibere, ou plus simple de renvoyer (statut, script_ou_erreur) comme le
    #    reste du projet ? A trancher avec lis_synthese/genere_voix (palier 5/6).
    try:
        dossier_documents.mkdir(parents=True, exist_ok=True)
        chemin = dossier_documents / document_name
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(reponse.strip())
        return "succes",chemin

    except Exception as e:
        return "echec", e

def synthese_conversation_vocale(repertoire_synthese: Path, nom_fichier_synthese, dossier_vocal: Path, nom_fichier_audio: str = "lecture_simple.wav") -> tuple[str, str]:
    """Lit a voix haute (une seule voix, Piper/Kokoro) le resume deja calcule d'une
    conversation - palier 5 (niveau 2) : un seul fichier audio continu pour tout le
    texte (decide le 2026-09-24 - pas de decoupage phrase par phrase comme la demo
    audio_local/lecture_simple.py, qui decoupait a la main pour montrer une lecture
    PROGRESSIVE, pas pratique ici). synthesize() accepte deja un texte de n'importe
    quelle longueur - un seul appel suffit, pas besoin de boucle.

    Args:
        repertoire_synthese : dossier contenant le fichier JSON de la conversation
            (LectureAssisteContexte/, deja existant, pas cree ici).
        nom_fichier_synthese : nom du fichier JSON de la conversation (ex:
            "LectureAssisteContext_{profil_id}_{id_conversation}.json").
        dossier_vocal : dossier ou ecrire le fichier audio (ex: voice/ de
            documents_{id_conversation}/), cree ici si besoin.
        nom_fichier_audio : nom du fichier audio a ecrire.

    Returns:
        (statut, chemin_ou_message) - meme convention que add_to_db/analyser_document :
        ("succes", chemin du fichier audio genere) ou ("echec", message d'erreur).
    """
    chemin_fichier_synthese = repertoire_synthese / nom_fichier_synthese

    # except FileNotFoundError resserre UNIQUEMENT autour de cette ouverture
    # (2026-09-24, corrige apres un test reel) - avant, un seul except englobait tout
    # le bloc, donc une FileNotFoundError venant de get_tts_provider() (modele TTS
    # manquant, bug distinct corrige le meme jour dans providers_tts.py) etait
    # signalee a tort comme "fichier conversation introuvable", alors que le JSON
    # existait bel et bien - message trompeur qui aurait fait chercher au mauvais
    # endroit.
    try:
        with open(chemin_fichier_synthese, "r", encoding="utf-8") as f:
            conversation_chargee = StructureConversation(**json.load(f))
    except FileNotFoundError:
        return "echec", f"fichier conversation introuvable : {chemin_fichier_synthese}"

    synthese_conversation = conversation_chargee.resume

    try:
        provider = get_tts_provider("fr")  # Piper par defaut en francais

        dossier_vocal.mkdir(parents=True, exist_ok=True)
        chemin_audio = dossier_vocal / nom_fichier_audio
        provider.synthesize(synthese_conversation, str(chemin_audio))
        return "succes", str(chemin_audio)

    except Exception as e:
        return "echec", str(e)



def genere_podcast(db, dossier_documents: Path, niveau_public="", ton="", points_particuliers="") -> tuple[str, str]:
    """Genere le podcast complet (script + audio a deux voix) pour un document -
    palier 6 (niveau 2). Reutilise genere_script_2voix (palier 4) : si le script
    existe deja pour cette conversation, il n'est pas regenere (evite un appel LLM
    redondant, meme principe que la persistance de `resume`) ; sinon il est genere
    a la volee avant de passer a l'audio.

    Args:
        db : le wrapper ChromeDb de la conversation active (pas db.db directement -
            db.db est extrait ici, comme dans main.py pour les autres appels).
        dossier_documents : dossier de la conversation (documents_{id_conversation}/)
            - le sous-dossier voice/ y est cree si besoin.
        niveau_public, ton, points_particuliers : preferences utilisateur pour le
            script (voir genere_script_2voix) - ignorees si le script existe deja.

    Returns:
        (statut, chemin_ou_message) - meme convention que add_to_db/analyser_document :
        ("succes", chemin du fichier audio genere) ou ("echec", message d'erreur).
    """
    dossier_voice = dossier_documents / "voice"
    script_file_name = "script_podcast.txt"
    audio_file_name = "podcast_deux_voix_output.wav"
    chemin_script = dossier_voice / script_file_name
    chemin_podcast = dossier_voice / audio_file_name

    dossier_voice.mkdir(parents=True, exist_ok=True)

    if chemin_podcast.is_file():
        return "succes", str(chemin_podcast)

    # Les deux branches (script deja la / a generer) convergent vers UN SEUL statut
    # commun, verifie une seule fois plus bas - evite de dupliquer l'appel Dia2 dans
    # chaque branche (piege discute avec thankgirl le 2026-09-25).
    if chemin_script.is_file():
        statut_script = "succes"
        # chemin_script deja correct (calcule ci-dessus), rien d'autre a faire.
    else:
        statut_script, chemin_script = genere_script_2voix(
            db.db, dossier_voice, script_file_name,
            niveau_public=niveau_public, ton=ton, points_particuliers=points_particuliers,
        )

    if statut_script != "succes":
        return "echec", f"echec generation script : {chemin_script}"

    # A partir d'ici, dans les DEUX cas, le script est garanti pret - un seul appel
    # Dia2. synthesize() attend le TEXTE du script, pas son chemin - il faut lire le
    # fichier avant (piege releve en audit : chemin_script etait passe tel quel).
    try:
        texte_script = Path(chemin_script).read_text(encoding="utf-8")
        provider = get_tts_provider("fr", engine="dia2")
        provider.synthesize(texte_script, str(chemin_podcast))
        return "succes", str(chemin_podcast)

    except Exception as e:
        return "echec", str(e)
