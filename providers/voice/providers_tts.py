# Interface commune pour les fournisseurs de synthese vocale (TTS) - meme principe que
# providers.py (LLM) utilise ailleurs dans le parcours : une interface stable, plusieurs
# implementations interchangeables (locales aujourd'hui, un fournisseur payant type
# ElevenLabs demain sans rien changer au code appelant).
from abc import ABC, abstractmethod
from pathlib import Path
import wave

# AUDIO_MODELS_DIR ancre sur __file__ (2026-09-24, meme principe que PROMPTS_DIR dans
# rag_chain.py/podcast_service.py et persist_directory dans vector_store_provider.py) -
# PAS des chemins relatifs nus ("kokoro-v1.0.onnx") comme avant, qui ne se resolvaient
# correctement que si le programme etait lance depuis audio_local/ (ou vivent ces
# fichiers de modele) - jamais le cas en usage reel (l'app est lancee depuis travail/).
# Bug reel trouve ce jour en testant palier 5 : get_tts_provider("fr") plantait avec
# FileNotFoundError, message trompeur cote appelant (voir podcast_service.py) qui
# laissait croire a un probleme de fichier de conversation.
# __file__ = .../travail/providers/voice/providers_tts.py -> 3x .parent = .../travail/
AUDIO_MODELS_DIR = Path(__file__).parent.parent.parent / "audio_local"


class TTSProvider(ABC):
    """Interface commune : toute implementation doit savoir transformer du texte en
    fichier audio. Le code appelant ne connait que cette methode, jamais les details
    d'un moteur precis (Kokoro, Piper, ou futur fournisseur externe)."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> None:
        ...


class KokoroProvider(TTSProvider):
    """Kokoro (kokoro-onnx) - modele local, une seule voix a la fois.
    22 langues dont l'anglais (tres bonne qualite) et le francais (qualite plus faible,
    voir AUDIO_LOCAL_MODE_OPERATOIRE.md)."""

    def __init__(self, voice: str, lang: str,
                 model_path: str = str(AUDIO_MODELS_DIR / "kokoro-v1.0.onnx"),
                 voices_path: str = str(AUDIO_MODELS_DIR / "voices-v1.0.bin")):
        from kokoro_onnx import Kokoro
        self._model = Kokoro(model_path, voices_path)
        self._voice = voice
        self._lang = lang

    def synthesize(self, text: str, output_path: str) -> None:
        import soundfile as sf
        audio, sample_rate = self._model.create(text, voice=self._voice, speed=1.0, lang=self._lang)
        sf.write(output_path, audio, sample_rate)


class Dia2Provider(TTSProvider):
    """Dia2 (Nari Labs) - modele local, dialogue a deux voix via les tags [S1]/[S2]
    directement dans le texte (pas deux appels separes, un seul texte/un seul
    fichier - rentre dans le meme contrat `synthesize(text, output_path)` que les
    autres fournisseurs, voir podcast_deux_voix.py deja teste). Exige un GPU CUDA
    (confirme present sur cette machine, RTX 5070 Laptop)."""

    def __init__(self, model_path: str = "nari-labs/Dia2-1B"):
        # model_path = identifiant de depot HuggingFace (pas un fichier local) -
        # Dia2.from_repo() le telecharge/le met en cache lui-meme, contrairement a
        # Kokoro/Piper qui chargent un fichier .onnx deja present sur disque. Ne
        # PAS le combiner avec AUDIO_MODELS_DIR (bug corrige le 2026-09-25) - ce
        # n'est pas un chemin local.
        from dia2 import Dia2, GenerationConfig, SamplingConfig

        self.dia = Dia2.from_repo(model_path, device="cuda", dtype="bfloat16")
        self.config = GenerationConfig(
            cfg_scale=2.0,
            audio=SamplingConfig(temperature=0.8, top_k=50),
            use_cuda_graph=False,
        )

    def synthesize(self, text: str, output_path: str) -> None:
        # dia.generate(..., output_wav=...) ecrit le fichier LUI-MEME (voir
        # podcast_deux_voix.py, deja teste le 2026-08-25 - aucun sf.write() apres
        # cet appel) - contrairement a Kokoro (create() renvoie un tableau audio
        # brut, sf.write() necessaire separement). output_path (pas un nom en dur)
        # pour que l'appelant choisisse reellement ou le fichier atterrit - bug
        # corrige le 2026-09-25 (ecrivait toujours vers le meme fichier fixe).
        self.dia.generate(text, config=self.config, output_wav=output_path, verbose=True)




class PiperProvider(TTSProvider):
    """Piper - modele local, tres peu de dependances (juste onnxruntime), voix plus
    mecanique mais fiable (pas de bafouillage constate en francais, contrairement a
    Kokoro sur cette langue)."""

    def __init__(self, model_path: str):
        from piper import PiperVoice
        self._voice = PiperVoice.load(model_path)

    def synthesize(self, text: str, output_path: str) -> None:
        with wave.open(output_path, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)




# Registre langue -> fournisseur par defaut. Pour ajouter une langue ou changer le
# moteur par defaut d'une langue existante : une seule ligne a modifier ici, rien
# ailleurs dans le code appelant.
def get_tts_provider(language: str, engine: str | None = None) -> TTSProvider:
    """
    language : "en" ou "fr" (a etendre au fil des besoins).
    engine : force un moteur precis ("kokoro" ou "piper") ; sinon, le meilleur
             disponible pour cette langue est choisi automatiquement.
    """
    if engine == "dia2":
        return Dia2Provider()  # dialogue a deux voix, independant de la langue
    
    if language == "en":
        return KokoroProvider(voice="af_heart", lang="en-us")

    if language == "fr":
        if engine == "kokoro":
            return KokoroProvider(voice="ff_siwis", lang="fr-fr")
        # Piper par defaut en francais : plus fiable (voir comparatif du 2026-08-25)
        return PiperProvider(str(AUDIO_MODELS_DIR / "fr_FR-siwis-medium.onnx"))

    raise ValueError(f"Langue non supportee: {language!r} (supportees: 'en', 'fr')")

