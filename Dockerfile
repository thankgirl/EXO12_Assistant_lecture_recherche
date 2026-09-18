# Dockerfile — deploiement niveau 1 (RAG/conversations) de l'exercice 12
#
# Construit le 2026-09-18. Ne prend QUE le niveau 1 (main.py + core/services/providers/
# interfaces/prompts/config/governance) - pas audio_local/ (niveau 2/3), qui a ses
# propres dependances lourdes (kokoro-onnx, piper-tts, dia2, torch CUDA...) sans rapport
# avec ce qui est deploye ici. Voir requirements-deploy.txt pour le detail du choix des
# dependances.

# python:3.11-slim, pas "alpine" : les wheels precompilees scientifiques (numpy,
# torch, scikit-learn) sont construites pour glibc, pas la musl libc d'alpine - les
# utiliser sous alpine forcerait une recompilation depuis les sources, beaucoup plus
# lente et parfois source d'echecs de build. 3.11 = meme version que le venv local du
# projet (SETUP_PROCEDURE.md - les wheels ML ne supportent pas encore les Python plus
# recents utilises par defaut sur la machine de developpement).
FROM python:3.11-slim

WORKDIR /app

# libmagic1 : dependance systeme requise par `unstructured` (detection de type de
# fichier pour UnstructuredExcelLoader, .xls) - absente de l'image de base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# requirements-deploy.txt copie seul, AVANT le reste du code : profite du cache de
# build par couches de Docker (voir DOCKER_MODE_OPERATOIRE.md, section 0.1) - tant que
# ce fichier ne change pas, Docker reutilise la couche d'installation deja construite
# au lieu de tout reinstaller a chaque modification de main.py.
COPY requirements-deploy.txt .

# torch en version CPU (pas la build CUDA "+cu128" du venv local - inutile et enorme
# sans GPU dans le conteneur) - installe separement, depuis l'index officiel PyTorch
# CPU, AVANT le reste des dependances (qui en ont besoin en transitif via
# sentence-transformers).
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Code de l'application - niveau 1 uniquement.
COPY main.py .
COPY core/ core/
COPY services/ services/
COPY providers/ providers/
COPY interfaces/ interfaces/
COPY prompts/ prompts/
COPY config/ config/
COPY governance/ governance/

# ANTHROPIC_API_KEY : jamais copiee/codee ici (GESTION_CLES_API.md,
# SETUP_PROCEDURE.md section 9) - fournie a l'execution via une variable
# d'environnement (`docker run -e ANTHROPIC_API_KEY=...`) ou un Secret OpenShift.

# OpenShift fait tourner les conteneurs avec un UID ALEATOIRE non-root par defaut
# (Security Context Constraints "restricted") - contrairement a un simple
# "docker run" en local (root par defaut ici, faute de USER precise), l'appli
# plante en PermissionError des qu'elle essaie de creer un dossier sous /app en
# conditions reelles OpenShift (ex: contexte_system/, LectureAssisteContexte/, crees
# a l'execution, pas presents dans l'image). Correctif standard Red Hat : donner au
# groupe root (GID 0, TOUJOURS present pour l'UID aleatoire d'OpenShift, quel qu'il
# soit) les memes droits que le proprietaire sur /app.
RUN chgrp -R 0 /app && chmod -R g=u /app

# Meme famille de piege, rencontre une 2e fois (2026-09-18) : l'UID aleatoire n'a
# aucune entree dans /etc/passwd, donc son "home" retombe sur "/" (racine du
# systeme, jamais inscriptible) - HuggingFace (telechargement du modele
# d'embeddings, providers/vector_store_provider.py) essaie d'y creer ~/.cache et
# plante en PermissionError sur '/.cache'. Fixer HOME sur un dossier deja ouvert
# en ecriture (/app, voir juste au-dessus) resout cette famille entiere de
# problemes (cache HuggingFace, pip, config diverses) d'un coup, plutot que de
# corriger chaque bibliotheque une par une au fil des erreurs.
ENV HOME=/app

# 3e symptome de la meme famille (2026-09-18), plus fondamental : l'UID aleatoire
# OpenShift n'a AUCUNE entree dans /etc/passwd (connu seulement au demarrage du
# pod, pas au moment du build) - torch plante avec "KeyError: getpwuid(): uid not
# found" des qu'une bibliotheque appelle getpass.getuser()/pwd.getpwuid(), meme
# avec HOME deja fixe. /etc/passwd doit etre inscriptible par le groupe root (GID 0)
# pour que le script de demarrage (entrypoint.sh) puisse y enregistrer l'UID
# assigne au lancement, avant que quoi que ce soit d'autre ne s'execute.
RUN chmod g=u /etc/passwd
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8501

# --server.address=0.0.0.0 : par defaut Streamlit n'ecoute que sur localhost DANS le
# conteneur - sans ca, inaccessible depuis l'exterieur meme avec le port publie.
# --server.headless=true : evite que Streamlit tente d'ouvrir un navigateur (inexistant
# dans le conteneur) au demarrage.
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
