# Trois noms qui se ressemblent, a ne pas confondre :
# - providers/         = ce DOSSIER (package Python, contient ce __init__.py)
# - LLM_providers.py    = le FICHIER a l'interieur, ou sont ecrites les vraies classes
# - LLMProvider/ClaudeProvider/OpenAIProvider = les CLASSES definies dans ce fichier
#
# Les lignes ci-dessous "remontent" ces classes au niveau du dossier providers/ -
# n'importe quel autre fichier peut alors faire `from providers import ClaudeProvider`
# sans avoir besoin de connaitre le nom exact du fichier (LLM_providers.py) ou elle
# est reellement ecrite. C'est le role de __init__.py : la "table des matieres" du
# package, pas de logique metier ici, juste des imports qui pointent vers l'existant.
#
# Ca fonctionne des lors que l'app est lancee depuis travail/ (streamlit run main.py)
# - c'est ce point de depart qui permet a Python de retrouver le dossier providers/.
from .LLM_providers import LLMProvider, ClaudeProvider, OpenAIProvider
from .vector_store_provider import VectorProvider, ChromeDb

__all__ = ["LLMProvider", "ClaudeProvider", "OpenAIProvider","VectorProvider", "ChromeDb"]

