# Assistant de lecture / recherche — RAG + conversations

Assistant Streamlit qui répond à des questions sur des documents (PDF, `.doc`, `.txt`,
`.xls`, `.csv`) en s'appuyant sur leur contenu réel (RAG), organisé en conversations
persistantes (une conversation = un fil de documents + un historique de questions/
réponses, isolé des autres conversations).

Extrait déployable d'un exercice pédagogique plus large (parcours d'apprentissage AI
Engineer) — ce dépôt ne contient que le code nécessaire à l'exécution de l'application,
pas les niveaux audio (TTS/STT) ni les documents pédagogiques du parcours d'origine.

## Fonctionnement

- **Contexte système** : une expertise déclarée une fois par profil (ex: "expert en
  IA"), sert de garde-fou pour juger la pertinence des documents analysés.
- **Conversations** : chacune a sa propre base vectorielle (Chroma) isolée, peut
  recevoir plusieurs documents au fil du temps, garde son historique de questions/
  réponses (mode chat) d'une session à l'autre.
- **Gestion du contexte long** : au-delà d'un certain volume d'historique, la
  conversation est automatiquement résumée, ou une recherche sélective sur les
  échanges passés remplace l'injection complète — au-delà d'un second seuil, une
  nouvelle conversation est requise.

## Lancer en local

```
pip install -r requirements-deploy.txt
streamlit run main.py
```

Nécessite une variable d'environnement `ANTHROPIC_API_KEY` (jamais commise dans ce
dépôt).

## Avec Docker

```
docker build -t assistant-lecture-rag:local .
docker run -d -p 8501:8501 -e ANTHROPIC_API_KEY="..." assistant-lecture-rag:local
```

## Architecture

- `main.py` — point d'assemblage (interface Streamlit).
- `services/` — logique métier (RAG, gestion des conversations, extraction de
  documents).
- `providers/` — fournisseurs LLM (Claude) et base vectorielle (Chroma), isolés
  derrière une interface commune.
- `core/` — validation des fichiers entrants, gestion de la taille du contexte.
- `interfaces/` — schémas de données (Pydantic).
- `prompts/` — prompts externalisés, versionnés.
- `config/`, `governance/` — paramètres et règles de validation, externalisés (jamais
  codés en dur).
