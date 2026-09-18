# Lancer avec : c:\Users\aelsa\Documents\git-practice\parcours_exercices\couche2_casse_repare\exercice_12_assistant_lecture_recherche\travail\venv\Scripts\python.exe -m streamlit run "c:\Users\aelsa\Documents\git-practice\parcours_exercices\couche2_casse_repare\exercice_12_assistant_lecture_recherche\travail\main.py"
"""
main.py — point d'assemblage unique (MOD-2), interface Streamlit

En lisant ce fichier, on doit voir toute l'architecture : la logique metier (RAG,
lecture/ecriture du contexte, traitement des PDF) vit dans services/ - ici, on
appelle juste ces fonctions et on affiche le resultat.

Trois notions distinctes a ne pas confondre (mises a jour le 2026-09-17) :
- `contexte` (l'expertise du SYSTEME, ex: "expert en IA") : defini une fois par
  profil_id, PAS optionnel pour analyser un document - sans lui, la verification de
  pertinence (analyser_document) n'a rien a comparer et ne rejetterait jamais rien.
  Voir contexte_system/README.md.
- une CONVERSATION (LectureAssisteMemoire) : un fil documentaire + historique de
  questions/reponses, titre, isole (sa propre collection Chroma) - on peut en avoir
  plusieurs par profil_id, en choisir une ou en creer une nouvelle a chaque lancement.
- `query` (la question de l'UTILISATEUR dans le chat) : optionnelle - si absente, on
  s'arrete a la synthese automatique des documents deja analyses.

Flux :
1. Contexte SYSTEME (expertise) : demande une seule fois (premier lancement), obligatoire
   avant de choisir/creer une conversation.
2. Choix de la conversation (menu deroulant) : une existante (recharge son historique) ou
   "+ Nouvelle conversation" (demande un titre).
3. Upload de document(s) sur la conversation active - reste possible a tout moment, pas
   seulement a la creation (plusieurs documents peuvent s'accumuler dans la meme
   conversation). Chaque upload declenche une verification "le(s) document(s)
   correspondent-ils au contexte ?" + synthese sur l'ensemble deja accumule.
4. Si au moins un document a ete juge pertinent (cette session ou lors d'une session
   precedente) : chat ouvert (st.chat_message/st.chat_input) pour poser des questions,
   historique conserve d'un lancement a l'autre.
"""
import os
import getpass
import json
import yaml
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from services import run_rag_chain, analyser_document, add_to_db, LectureAssisteMemoire, context_system_read, context_system_write
from providers import ClaudeProvider, ChromeDb  # deja reexportee par providers/__init__.py, verifie
from core import context_management
from interfaces.schemas import StructureContext, StructureConversation
from langchain_core.prompts import ChatPromptTemplate            # gabarit de prompt avec des "trous" à remplir
from langchain_core.output_parsers import StrOutputParser        # extrait le texte brut de la réponse
from langchain_core.runnables import RunnablePassthrough         # laisse une valeur passer sans la modifier

load_dotenv()  # lit le fichier .env du dossier courant et remplit os.environ avec son contenu

# Types de fichiers acceptes par st.file_uploader plus bas - lus depuis
# input_guardrails.yaml (meme fichier que core/file_validator.py, seule source de
# verite - CTX-3) plutot que redupliques en dur ici. Bug trouve le 2026-09-17 : le
# uploader etait fige sur type=["pdf"] (reste de la toute premiere version de
# l'exercice), alors que .doc/.txt/.xls/.csv etaient deja acceptes cote guardrails/
# document_service.py - un .doc valide etait donc rejete avant meme d'atteindre la
# validation. st.file_uploader attend les extensions SANS le point (".doc" -> "doc").
GUARDRAILS_DIR = Path(__file__).parent / "governance" / "guardrails"
with open(GUARDRAILS_DIR / "input_guardrails.yaml", "r", encoding="utf-8") as f:
    guardrails_param = yaml.safe_load(f)
TYPES_AUTORISES = [ext.lstrip(".") for ext in guardrails_param["fichiers_document"]["types_autorises"]]


# ============================================================
# TODO (mis a jour le 2026-09-17) :
# 1. db = ChromeDb(...) -> fait le 2026-09-18 : @st.cache_resource sur get_db(),
#    collection_name en parametre (voir juste avant main()). Revele en deployant sur
#    OpenShift (pod limite en ressources) : sans ca, le modele d'embeddings etait
#    recharge a chaque interaction, lenteurs/blocages intermittents.
# 2. CONVERSATIONS -> fait le 2026-09-17 : menu deroulant (nouvelle/existante),
#    ChromeDb initialisee APRES le choix de la conversation (collection_name derive de
#    id_conversation, isole chaque conversation dans sa propre collection Chroma),
#    mode chat (st.chat_message/st.chat_input) a la place du text_area/bouton. Reste
#    ouvert : pas de methode dediee cote contexte_service.py pour charger UNE
#    conversation existante (lister_conversation liste, mais rien ne recharge) - fait
#    ici en lisant le fichier JSON directement (memes chemin/convention que
#    contexte_service.py) le temps que ce soit fait proprement cote service (MOD-1).
# 3. Rebranchement context_manager.py -> fait le 2026-09-17 : chemin_fichier_conversation
#    calcule sans condition, context_management(...) appele avant run_rag_chain,
#    historique_contexte injecte dans le prompt (rag_chain.py/qr-rag.yaml etendus).
#    depassement_taille_max bloque et force une nouvelle conversation (return DANS le
#    if, deux tentatives precedentes l'avaient mal place). Bug corrige au passage :
#    premiere question d'une conversation neuve (echanges=[]) faisait planter
#    count_tokens sur une chaine vide (BadRequestError Anthropic) - court-circuite
#    dans context_manager.py.
# 4. Verifier les signatures reelles de add_to_db/analyser_document/run_rag_chain
#    (services/) correspondent toujours a ce que main.py leur passe - plusieurs ont
#    change depuis la derniere fois que main.py a ete touche.
# 5. (reporte) add_to_db stocke le document dans db (chunking + embeddings, coute cher)
#    AVANT qu'analyser_document verifie sa pertinence au contexte - si non pertinent, le
#    travail est perdu et les chunks restent dans db (rien ne les retire). Ordre correct :
#    separer add_to_db en extraction (texte brut) + stockage (chunks/embeddings),
#    verifier la pertinence sur le texte brut AVANT le stockage - touche
#    document_service.py ET rag_chain.py, pas juste main.py.
# 6. (reporte - option propre choisie pour plus tard) db.db utilise partout ci-dessous
#    (add_to_db/analyser_document/run_rag_chain) - correctif rapide, expose l'interieur
#    du wrapper ChromeDb. Mieux : ajouter a ChromeDb des methodes qui redirigent vers
#    self.db (.get(), .as_retriever(), .add_documents()), pour que le reste du code ne
#    connaisse jamais que ChromeDb, jamais Chroma directement (MOD-3).
# (garde-fou sur la taille du corpus de documents d'une conversation - envisage un temps
# comme point 7 ici le 2026-09-17, deplace le meme jour vers SUJETS_APPROFONDIR_PLUS_TARD.md :
# ne bloque pas la fin du niveau 1, pas a melanger avec le TODO actif)
# ============================================================

# TODO #1 (fait le 2026-09-18) : sans @st.cache_resource, ChromeDb() etait reconstruite
# a CHAQUE interaction (Streamlit relance tout main() a chaque clic) - donc le modele
# d'embeddings etait recharge en memoire depuis zero a chaque fois. Invisible en local
# (CPU/RAM larges), mais source de lenteurs/blocages intermittents sur le pod OpenShift
# limite en ressources (constate le 2026-09-18 : "ca analyse, puis rien" au 1er essai,
# fonctionne au 2e - le rechargement du modele prenait plus de temps que ce que
# l'utilisateur attendait avant de recliquer). collection_name en parametre de la
# fonction cachee (pas juste sur ChromeDb()) : Streamlit met en cache PAR JEU DE
# PARAMETRES - une conversation differente (collection_name different) reconstruit
# bien sa propre instance, mais revenir sur une conversation deja visitee reutilise
# l'instance deja chargee, sans recharger le modele.
@st.cache_resource
def get_db(collection_name: str) -> ChromeDb:
    return ChromeDb(collection_name=collection_name)


def main():
    st.set_page_config(page_title="Assistant apprentissage", page_icon="🔄")
    st.header("Career Change Insight Retrieval System")

    api_key=os.getenv("ANTHROPIC_API_KEY", "")
    # db = ChromeDb() deplace plus bas (2026-09-17) : collection_name est maintenant
    # obligatoire et depend de la conversation active, choisie plus loin dans le script -
    # impossible de construire la base avant de savoir quelle conversation est ouverte.

    # profil_id : identifiant simple pour l'instant (nom de session Windows) - meme
    # principe que vector_store_provider.py (2026-08-27), un vrai multi-utilisateur
    # est un exercice a part, pas le coeur de celui-ci.
    profil_id = getpass.getuser()

    # --- Contexte SYSTEME (expertise) : verifie si deja defini pour ce profil_id ---
    # Different de "query" plus bas : celui-ci definit ce que le systeme est cense
    # savoir/verifier, pas une question ponctuelle de l'utilisateur. Different aussi des
    # conversations (LectureAssisteMemoire) : une seule expertise par profil_id, pas une
    # par document/recherche - voir contexte_system/README.md.
    contexte = context_system_read(profil_id)  # "" si le fichier n'existe pas encore

    if contexte:
        st.info(f":bulb: Expertise système actuelle : **{contexte}**")
        # st.session_state.changer_expertise : survit aux reruns Streamlit, necessaire
        # car le clic sur "Changer d'expertise" et le clic sur "Enregistrer" sont deux
        # interactions separees (deux reruns du script).
        if st.button("Changer d'expertise"):
            st.session_state.changer_expertise = True
        if st.session_state.get("changer_expertise"):
            nouvelle_expertise = st.text_area(
                ":bulb: Nouvelle expertise système",
                placeholder="ex: RAG IA",
            )
            if nouvelle_expertise and st.button("Enregistrer la nouvelle expertise"):
                context_system_write(profil_id, nouvelle_expertise)
                contexte = nouvelle_expertise
                st.session_state.changer_expertise = False
                st.success("Expertise mise à jour — écrase l'ancienne.")
    else:
        contexte = st.text_area(
            ":bulb: Dans quel domaine souhaitez-vous développer votre expertise ?",
            placeholder="ex: RAG IA",
        )
        if contexte:
            context_system_write(profil_id, contexte)
            st.success("Expertise enregistrée — elle ne sera plus redemandée au prochain lancement.")

    st.markdown("---")

    if not contexte:
        st.warning("Définissez d'abord le contexte d'apprentissage ci-dessus avant de choisir une conversation.")
        return  # rien de plus a afficher tant que le contexte systeme n'est pas defini

    # --- CONVERSATIONS (2026-09-17) : choisir/creer AVANT d'initialiser ChromeDb ---
    # collection_name (donc ChromeDb) depend de id_conversation - impossible de
    # construire la base avant de savoir quelle conversation est active.
    memoire = LectureAssisteMemoire(profil_id)
    conversations_existantes = memoire.lister_conversation()  # [(titre, id_conversation), ...]

    NOUVELLE = "+ Nouvelle conversation"
    options = [NOUVELLE] + [titre for titre, _ in conversations_existantes]

    # Selection en attente (posee juste apres la creation d'une nouvelle conversation,
    # voir plus bas) : IMPOSSIBLE d'ecrire directement sur st.session_state d'un widget
    # deja instancie DANS LE MEME RUN (StreamlitAPIException, rencontree le 2026-09-17,
    # meme en ecrivant juste avant un st.rerun()) - d'ou cette variable intermediaire,
    # consommee ICI, AVANT que st.selectbox() ne soit appele plus bas.
    selection_en_attente = st.session_state.pop("conversation_a_selectionner", None)
    index_par_defaut = options.index(selection_en_attente) if selection_en_attente in options else 0

    choix_conversation = st.selectbox("Conversation", options, index=index_par_defaut)

    if choix_conversation == NOUVELLE:
        nouveau_titre = st.text_input("Titre de la nouvelle conversation", placeholder="ex: Papier sur le RAG")
        st.caption("Tape un titre, puis clique sur le bouton ci-dessous — appuyer sur Entrée ne suffit pas.")
        if nouveau_titre and st.button("Créer la conversation"):
            # creer_conversation() cree le fichier ET pose memoire.id_conversation -
            # echanges=[] : une conversation toute neuve n'a encore aucun echange.
            memoire.creer_conversation(titre=nouveau_titre, resume="", echanges=[])
            st.session_state.id_conversation = memoire.id_conversation
            st.session_state.titre_conversation = nouveau_titre
            st.session_state.echanges = []
            # Efface les anciens indicateurs (pertinence/synthese) d'une conversation
            # precedente - sinon une conversation neuve heriterait par erreur du
            # resultat d'analyse de la conversation quittee (meme cles de session_state,
            # partagees pour l'instant entre toutes les conversations).
            st.session_state.document_pertinent = False
            st.session_state.pop("synthese", None)
            # Sans cette ligne (bug trouve le 2026-09-17) : apres le rerun, le menu
            # deroulant restait affiche sur "+ Nouvelle conversation" (sa valeur
            # precedente, toujours valide puisque toujours dans options) au lieu de
            # basculer sur la conversation qui vient d'etre creee - ecran identique
            # a avant le clic, aucune confirmation visible. Cette variable (pas la
            # cle du widget lui-meme, voir plus haut) sera lue au prochain run, AVANT
            # que le selectbox soit instancie, pour lui donner le bon index par defaut.
            st.session_state.conversation_a_selectionner = nouveau_titre
            st.toast(f"Conversation « {nouveau_titre} » créée !", icon="✅")
            st.rerun()
        else:
            return  # pas de conversation active tant que le titre n'est pas valide

    else:
        # Conversation existante choisie dans le menu - retrouver son id_conversation
        # (lister_conversation() renvoie (titre, id_conversation), pas juste le titre -
        # necessaire pour reconstruire le nom exact du fichier).
        id_choisi = next(id_c for titre, id_c in conversations_existantes if titre == choix_conversation)

        if st.session_state.get("id_conversation") != id_choisi:
            # Changement de conversation (ou premier chargement de la page) - recharge
            # son historique depuis son fichier JSON. Pas de methode dediee cote
            # contexte_service.py pour lire UNE conversation existante (lister_conversation
            # liste les titres/id, actualiser_conversation ecrit, aucune ne "charge" pour
            # relire l'ecran) - fait ici en attendant, meme convention de nom de fichier
            # que contexte_service.py (LectureAssisteContext_{profil_id}_{id}.json).
            chemin_fichier = memoire.context_path / f"LectureAssisteContext_{profil_id}_{id_choisi}.json"
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                conversation_chargee = StructureConversation(**json.load(f))
            st.session_state.id_conversation = id_choisi
            st.session_state.titre_conversation = conversation_chargee.titre
            st.session_state.echanges = conversation_chargee.echanges
            # Synthese stockee (2026-09-17) : `resume` (champ StructureConversation,
            # jamais rempli avant aujourd'hui) porte maintenant la derniere synthese
            # calculee - la relire ici evite de rappeler le LLM juste pour rouvrir une
            # conversation qui a deja ete analysee. Si resume est vide (jamais analysee,
            # ou pas encore de document), on retombe sur "a analyser".
            if conversation_chargee.resume:
                st.session_state.document_pertinent = True
                st.session_state.synthese = conversation_chargee.resume
            else:
                st.session_state.document_pertinent = False
                st.session_state.pop("synthese", None)

    # --- A partir d'ici, une conversation est active : ChromeDb isolee par conversation ---
    # get_db() (pas ChromeDb() directement, depuis le 2026-09-18) : mise en cache par
    # collection_name, voir la fonction juste avant main() pour le detail complet.
    db = get_db(f"conversation_{st.session_state.id_conversation}")

    # Chemin du fichier JSON de la conversation active - calcule sans condition ici
    # (pas seulement dans la branche de rechargement ci-dessus) pour etre disponible
    # partout plus bas (context_management, sauvegarde de la synthese) meme quand
    # id_conversation n'a pas change entre deux reruns (piege deja rencontre une fois
    # avec chemin_fichier pour context_management - voir TODO #3 plus haut).
    chemin_fichier_conversation = memoire.context_path / f"LectureAssisteContext_{profil_id}_{st.session_state.id_conversation}.json"

    # Dossier de stockage des documents sources de cette conversation (2026-09-17) -
    # add_to_db y deplace chaque fichier une fois indexe, au lieu de le supprimer,
    # pour pouvoir le retrouver des semaines plus tard (une conversation sans son
    # document d'origine n'est pas exploitable). Meme dossier parent que les fichiers
    # JSON de conversation (LectureAssisteContexte/), deja hors suivi git.
    dossier_documents = memoire.context_path / f"documents_{st.session_state.id_conversation}"

    st.markdown(f"### 💬 {st.session_state.titre_conversation}")

    # --- Upload : reste accessible tant que la conversation est ouverte (2026-09-17) -
    # plusieurs documents peuvent etre ajoutes a la meme conversation au fil du temps,
    # pas seulement a sa creation. Chroma persiste sur disque : rouvrir cette conversation
    # (meme collection_name) retrouve les documents deja ajoutes, add_to_db en ajoute
    # d'autres sans les effacer.
    # key=f"uploader_{id_conversation}" (bug trouve le 2026-09-17) : sans cle propre a
    # la conversation active, Streamlit garde les fichiers selectionnes d'une
    # conversation a l'autre (meme identite de widget) - rien de mal cote donnees tant
    # qu'on ne clique pas "Analyser" dans cet etat, mais le clic aurait reellement
    # ajoute les fichiers de l'ancienne conversation dans la collection de la nouvelle.
    # Changer de conversation change la cle -> Streamlit reinitialise le widget a vide.
    pdf_docs = st.file_uploader(
        "Télécharger un document pour cette conversation",
        type=TYPES_AUTORISES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.id_conversation}",
    )

    if st.button("Analyser le document"):
        if not pdf_docs:
            st.warning("Téléversez un document svp")
        else:
            with st.spinner("Analyse du document..."):
                # db.db (pas db) : add_to_db/analyser_document/run_rag_chain
                # attendent un vrai objet Chroma (.add_documents(), .get(),
                # .as_retriever()) - db est notre wrapper ChromeDb (MOD-3), qui
                # n'expose pas ces methodes lui-meme (voir TODO #6).
                statut, resultat = add_to_db(pdf_docs, db.db, dossier_documents=dossier_documents)
                if statut != "success":
                    st.error(":file_folder: le document ne respecte pas les règles")
                    st.error(resultat)
                    st.session_state.document_pertinent = False
                else:
                    # Verifie la pertinence par rapport au contexte ET produit la
                    # synthese en un seul appel - sur TOUT le contenu de la collection
                    # (decide le 2026-09-17 : coherent avec le RAG plus bas qui pioche
                    # deja dans l'ensemble des documents de la conversation, pas
                    # document par document. Cout croissant avec le nombre de
                    # documents - voir TODO #7, garde-fou pas encore construit).
                    pertinent, message = analyser_document(db.db, contexte)
                    st.session_state.document_pertinent = pertinent
                    st.session_state.synthese = message
                    if pertinent:
                        # Persiste la synthese dans le champ `resume` du fichier JSON
                        # de la conversation (2026-09-17) - pour la relire au lieu de
                        # rappeler le LLM la prochaine fois que cette conversation est
                        # rouverte (voir plus haut, branche de rechargement). Lecture-
                        # modification-ecriture directe, meme raccourci assume que pour
                        # le rechargement (pas de methode dediee cote contexte_service.py).
                        with open(chemin_fichier_conversation, "r", encoding="utf-8") as f:
                            conversation_a_jour = StructureConversation(**json.load(f))
                        conversation_a_jour.resume = message
                        with open(chemin_fichier_conversation, "w", encoding="utf-8") as f:
                            json.dump(conversation_a_jour.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
                        st.success(":file_folder: Document ajouté et analysé avec succès !")
                    else:
                        st.warning("Ce document ne semble pas correspondre au contexte déclaré.")

    if st.session_state.get("document_pertinent"):
        # Lien vers les documents sources stockes (2026-09-17) - AVANT la synthese
        # (bug remonte le meme jour : place apres, il etait perdu de vue des que la
        # synthese etait longue - toujours visible en premiere ligne maintenant).
        # Liste simplement le contenu de dossier_documents plutot que de demander a
        # add_to_db de renvoyer la liste explicitement (moins de changement de
        # signature, meme resultat : ce que add_to_db vient d'y deplacer y est deja).
        if dossier_documents.exists():
            fichiers_stockes = sorted(dossier_documents.glob("*"))
            if fichiers_stockes:
                st.caption("📎 Document(s) source de cette conversation :")
                for fichier in fichiers_stockes:
                    st.caption(f"`{fichier}`")
        # Repliable (2026-09-17, demande de thankgirl) : une synthese longue forcait a
        # tout parcourir avant d'atteindre l'historique/le chat plus bas - repliee par
        # defaut, elle reste consultable sans dominer la page. Note : st.chat_input()
        # plus bas reste de toute facon epingle en bas de page par Streamlit (utilise
        # hors de tout conteneur), donc ceci n'affecte que la LECTURE de la synthese,
        # pas la possibilite de taper une question.
        with st.expander("Synthèse des documents de cette conversation", expanded=False):
            st.write(st.session_state.synthese)

    st.markdown("---")

    # --- Mode chat (2026-09-17) : remplace le text_area/bouton "Submit" -----------
    # Affiche l'historique existant (echanges rechargeacs ou deja poses cette session)
    # facon messagerie, puis st.chat_input() pour continuer - meme principe que
    # StructureContext (role/texte/date), un echange = un message affiche.
    for echange in st.session_state.get("echanges", []):
        with st.chat_message(echange.role):
            st.write(echange.texte)

    # Autorise a poser une question si un document a deja ete juge pertinent CETTE
    # session, OU si la conversation rechargee a deja un historique (preuve qu'un
    # document a ete valide pertinent lors d'une session precedente - pas besoin de
    # re-analyser un document juste pour continuer a discuter).
    peut_discuter = st.session_state.get("document_pertinent") or bool(st.session_state.get("echanges"))

    if not peut_discuter:
        st.info("Analysez d'abord un document pertinent pour pouvoir discuter dans cette conversation.")
    else:
        question = st.chat_input("Une question sur le(s) document(s) de cette conversation ?")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.spinner("Thinking..."):
                # on verifie la taille du contexte de cette conversation, on applique
                # des mesures selon le quota avant application du RAG
                statut, message = context_management(chemin_fichier_conversation, question)

                if statut == "depassement_taille_max":
                    # return A L'INTERIEUR de ce if (bug corrige le 2026-09-17, deux
                    # tentatives precedentes l'avaient mis soit dans un else accroche
                    # au mauvais if, soit inconditionnel) - bloque uniquement ce cas
                    # precis, sans empecher les 3 autres statuts de repondre.
                    nouveau_titre = st.text_input("Limite contexte depassée vous devez créer une nouvelle conversation", placeholder="ex: Papier sur le RAG")
                    st.caption("Tape un titre, puis clique sur le bouton ci-dessous — appuyer sur Entrée ne suffit pas.")
                    if nouveau_titre and st.button("Créer la conversation"):
                        # creer_conversation() cree le fichier ET pose memoire.id_conversation -
                        # echanges=[] : une conversation toute neuve n'a encore aucun echange.
                        memoire.creer_conversation(titre=nouveau_titre, resume="", echanges=[])
                        st.session_state.id_conversation = memoire.id_conversation
                        st.session_state.titre_conversation = nouveau_titre
                        st.session_state.echanges = []
                        st.session_state.document_pertinent = False
                        st.session_state.pop("synthese", None)
                        st.session_state.conversation_a_selectionner = nouveau_titre
                        st.toast(f"Conversation « {nouveau_titre} » créée !", icon="✅")
                        st.rerun()
                    return  # bloque : pas de reponse tant qu'une nouvelle conversation n'est pas creee

                # Atteint uniquement si statut != "depassement_taille_max" (par
                # elimination : contexte_resume/contexte_pertinent/contexte_normal) -
                # plus besoin de re-tester statut ici, le return ci-dessus a deja
                # filtre le seul cas ou on ne doit pas repondre.
                reponse = run_rag_chain(query=question, db=db.db, contexte_expertise=contexte, historique_contexte=message)
            with st.chat_message("assistant"):
                st.write(reponse)

            nouveaux_echanges = [
                StructureContext(sujet=st.session_state.titre_conversation, role="user", texte=question, date=datetime.now()),
                StructureContext(sujet=st.session_state.titre_conversation, role="assistant", texte=reponse, date=datetime.now()),
            ]
            st.session_state.echanges.extend(nouveaux_echanges)
            memoire.actualiser_conversation(
                id_conversation=st.session_state.id_conversation,
                profil_id=profil_id,
                echangesLLM=nouveaux_echanges,
            )

    # --- Zone de test (2026-09-04) : core/context_manager.py, les 3 paliers ---
    # Mise en commentaire le 2026-09-17 (demande de thankgirl) : on teste maintenant
    # context_manager directement via de vraies conversations (ex: "qualite des
    # donnees LLM" surchargee manuellement au-dela du seuil de depassement), ce
    # panneau avec ses tailles de fichiers simulees et sa question sur la tarte aux
    # pommes n'est plus necessaire pour l'instant - code garde tel quel, pas
    # supprime, au cas ou utile a nouveau plus tard.
    # st.markdown("---")
    # with st.expander("🧪 Tester context_manager (3 paliers simulés)"):
    #     SIMULATIONS_DIR = Path(__file__).parent / "simulations_contexte_manager"
    #     paliers_test = {
    #         "Palier 1 — résumé progressif (~170k tokens)": "palier_1_resume.json",
    #         "Palier 2 — recherche sélective (~235k tokens)": "palier_2_recherche_selective.json",
    #         "Palier 3 — nouvelle conversation (~285k tokens)": "palier_3_nouvelle_conversation.json",
    #     }
    #     choix = st.radio("Conversation simulée à tester", list(paliers_test.keys()))
    #
    #     # Palier 2 (recherche selective) : le fixture melange 8 echanges "marqueurs"
    #     # (recette de tarte aux pommes, sujet totalement different du reste) parmi le
    #     # contenu majoritaire - pose une question sur les pommes et verifie que le
    #     # resultat retourne bien CES echanges-la (preuve visuelle que la selection
    #     # suit la question, pas un choix arbitraire). Constate en testant le
    #     # 2026-09-04 : avec un contenu homogene partout, impossible de voir la
    #     # difference entre "ca marche" et "ca renvoie n'importe quoi".
    #     if choix.startswith("Palier 2"):
    #         st.caption(
    #             "💡 Ce fixture contient 8 échanges sur un sujet totalement différent "
    #             "(recette de tarte aux pommes) mêlés au reste — pose une question sur "
    #             "les pommes pour vérifier que la recherche les retrouve bien, et pas "
    #             "des échanges au hasard."
    #         )
    #     query_par_defaut = {
    #         "Palier 2 — recherche sélective (~235k tokens)": "À quelle température et combien de temps cuit la tarte aux pommes ?",
    #     }
    #     query_test = st.text_input(
    #         "Question de test (passée à context_management)",
    #         value=query_par_defaut.get(choix, "Résume les points essentiels de cette conversation."),
    #     )
    #     if st.button("Lancer le test"):
    #         chemin_fixture = SIMULATIONS_DIR / paliers_test[choix]
    #         with st.spinner("Appel de context_management..."):
    #             statut, message = context_management(str(chemin_fixture), query_test)
    #         st.write(f"**Statut renvoyé :** `{statut}`")
    #         st.write("**Message / action renvoyé :**")
    #         st.write(message)


if __name__ == "__main__":
    main()
