#!/bin/sh
# entrypoint.sh — support des UID arbitraires OpenShift (Security Context Constraints)
#
# L'UID assigne par OpenShift au demarrage du pod n'existe dans AUCUNE entree
# /etc/passwd construite a l'avance (connu seulement a l'execution, pas au build).
# Toute bibliotheque qui appelle getpass.getuser()/pwd.getpwuid() (ex:
# torch._inductor.runtime.cache_dir_utils) plante sinon avec
# "KeyError: getpwuid(): uid not found" - meme avec $HOME deja fixe sur un dossier
# inscriptible (voir Dockerfile, ENV HOME=/app). On enregistre l'UID courant dans
# /etc/passwd avant de lancer la vraie commande, seulement s'il n'y est pas deja
# (ex: execution normale en root ou UID connu, hors OpenShift).
if ! getent passwd "$(id -u)" > /dev/null 2>&1; then
    echo "streamlit:x:$(id -u):0:utilisateur dynamique OpenShift:/app:/sbin/nologin" >> /etc/passwd
fi

exec "$@"
