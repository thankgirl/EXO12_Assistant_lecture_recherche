"""
providers/LLM_providers.py — fournisseurs LLM isoles derriere une interface (MOD-3)

La logique metier (services/, rag_service.py a venir) ne connaitra que le contrat
LLMProvider (init + generate) - jamais LangChain directement. Changer de fournisseur =
ajouter une classe ici, zero ligne a changer dans services/.
"""
from abc import ABC, abstractmethod  # module standard Python pour definir un "contrat"
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class LLMProvider(ABC):
    """
    Le CONTRAT commun a tous les fournisseurs LLM - pas une classe qu'on utilise
    directement, juste la liste des methodes que chaque fournisseur DOIT avoir.
    'ABC' = Abstract Base Class : Python empeche de creer un objet LLMProvider() tout
    seul (ça n'aurait pas de sens, generate() n'a pas de corps ici) - on doit passer
    par une sous-classe concrete comme ClaudeProvider, qui elle fournit le vrai code.

    @abstractmethod = oblige chaque sous-classe a definir generate(), sinon Python
    refuse de la laisser s'instancier (erreur au demarrage, pas en pleine utilisation).
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        # **kwargs = "keyword arguments" : capture n'importe quel(s) parametre(s)
        # nomme(s) supplementaire(s) qu'on pourrait vouloir passer plus tard
        # (ex: generate(prompt, temperature=0.5)) SANS devoir changer la signature
        # de la methode ni celle de toutes les sous-classes. Optionnel a utiliser
        # pour l'instant - sert juste a ne pas se bloquer si un besoin arrive plus
        # tard (ex: ajuster la temperature au cas par cas).
        ...


class ClaudeProvider(LLMProvider):
    """Fournisseur concret pour Claude (Anthropic), via LangChain."""

    def __init__(self, api_key: str, model: str):
        # __init__ est appele UNE SEULE FOIS, au moment ou on cree l'objet :
        #     provider = ClaudeProvider(api_key="...", model="claude-sonnet-5")
        # Il configure l'objet (ici : prepare le client ChatAnthropic) et le stocke
        # dans self._model pour que generate() puisse s'en servir plus tard.
        # api_key recu en PARAMETRE (pas lu ici via os.getenv) : c'est le bon reflexe
        # MOD-2 - le composant reçoit sa dependance (la cle), il ne va pas la
        # chercher lui-meme dans l'environnement. C'est main.py (l'assemblage) qui
        # lira os.getenv("ANTHROPIC_API_KEY") et le passera ici.
        # PAS de temperature ici (retire le 2026-08-28) : les modeles Claude actuels
        # (dont claude-sonnet-5) refusent ce parametre - erreur 400 "temperature is
        # deprecated for this model", confirmee en testant la nouvelle cle API.
        self._model = ChatAnthropic(
            model=model,
            api_key=api_key,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # generate() est appele CHAQUE FOIS qu'on veut une reponse, potentiellement
        # plusieurs fois avec le meme objet provider deja configure :
        #     reponse1 = provider.generate("premiere question")
        #     reponse2 = provider.generate("deuxieme question")
        # .invoke(prompt) = methode LangChain qui envoie le prompt et attend la
        # reponse. .content = extrait le texte brut de l'objet reponse renvoye -
        # SAUF que claude-sonnet-5 a le "adaptive thinking" actif par defaut (meme
        # sans le demander) : sa reponse contient alors PLUSIEURS blocs (reflexion +
        # texte), et .content devient une LISTE de blocs au lieu d'une chaine.
        # Trouve/confirme le 2026-08-30 en testant l'app pour de vrai. On ne garde
        # que le texte, peu importe le nombre de blocs.
        contenu = self._model.invoke(prompt).content
        if isinstance(contenu, str):
            return contenu
        return "".join(
            bloc["text"] for bloc in contenu
            if isinstance(bloc, dict) and bloc.get("type") == "text"
        )

    def count_tokens (self, prompt: str, **kwargs) -> int:
        #compte le nombre de tokens en entrée et en sortie

        from anthropic import Anthropic

        client = Anthropic()
        resp = client.messages.count_tokens(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.input_tokens


class OpenAIProvider(LLMProvider):
    """Fournisseur concret pour OpenAI (GPT), via LangChain - meme structure que Claude."""

    def __init__(self, api_key: str, model: str):
        self._model = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.2,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        return self._model.invoke(prompt).content
