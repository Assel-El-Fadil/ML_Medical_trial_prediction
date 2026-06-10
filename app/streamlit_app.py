"""
app/streamlit_app.py
Interface Streamlit — Clinical Trial Completion Predictor
Accessible par un non-technicien.
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Clinical Trial Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personnalise
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Palette generale */
        [data-testid="stAppViewContainer"] { background: #0f1117; }
        [data-testid="stSidebar"] { background: #1a1d2e; }

        /* Titres */
        h1 { color: #e8eaf6; font-weight: 800; }
        h2, h3 { color: #c5cae9; }

        /* Cartes de resultat */
        .result-card {
            padding: 1.5rem 2rem;
            border-radius: 14px;
            margin-top: 1rem;
            text-align: center;
        }
        .result-abandon {
            background: linear-gradient(135deg, #4a0000 0%, #7f0000 100%);
            border: 2px solid #ef5350;
        }
        .result-complete {
            background: linear-gradient(135deg, #003300 0%, #1b5e20 100%);
            border: 2px solid #66bb6a;
        }
        .result-title { font-size: 1.8rem; font-weight: 800; margin-bottom: .5rem; }
        .result-proba { font-size: 2.8rem; font-weight: 900; }
        .result-conf  { font-size: 1rem; opacity: .85; margin-top: .4rem; }

        /* Badge confiance */
        .badge {
            display: inline-block;
            padding: .25rem .75rem;
            border-radius: 999px;
            font-size: .85rem;
            font-weight: 700;
            margin-top: .5rem;
        }
        .badge-high   { background:#1b5e20; color:#a5d6a7; }
        .badge-medium { background:#e65100; color:#ffcc80; }
        .badge-low    { background:#4a148c; color:#e1bee7; }

        /* Metric cards */
        .metric-box {
            background: #1e2130;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            text-align: center;
            border: 1px solid #2d3250;
        }
        .metric-val { font-size: 1.8rem; font-weight: 800; color: #7c83fd; }
        .metric-lbl { font-size: .8rem; color: #9e9e9e; margin-top: .2rem; }

        /* Bouton principal */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #5c6bc0, #7c83fd);
            color: white;
            border: none;
            border-radius: 8px;
            padding: .6rem 2rem;
            font-weight: 700;
            font-size: 1rem;
            transition: opacity .2s;
            width: 100%;
        }
        div[data-testid="stButton"] > button:hover { opacity: .85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/test-tube.png",
        width=70,
    )
    st.markdown("## Trial Predictor")
    st.caption("Prediction d'issue d'essai clinique")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Prediction unitaire", "Prediction en lot", "Informations modele"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Statut API
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok and r.json().get("model_loaded"):
            st.success("API connectee")
        else:
            st.warning("API sans modele")
    except Exception:
        st.error("API inaccessible")

# ===========================================================================
# PAGE 1 — PREDICTION UNITAIRE
# ===========================================================================
if page == "Prediction unitaire":
    st.title("Prediction unitaire")
    st.markdown(
        "Renseignez les caracteristiques de l'essai clinique pour obtenir "
        "une prediction de son issue (abandon ou completion)."
    )
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Identification")
            phase = st.selectbox(
                "Phase",
                ["PHASE1", "PHASE2", "PHASE3", "PHASE4",
                 "PHASE1/PHASE2", "PHASE2/PHASE3", "NA", "EARLY_PHASE1"],
                index=1,
            )
            sponsor_type = st.selectbox(
                "Type de sponsor",
                ["INDUSTRY", "NIH", "FED", "OTHER", "INDIV", "NETWORK", "UNKNOWN"],
            )
            intervention_type = st.selectbox(
                "Type d'intervention",
                ["DRUG", "BIOLOGICAL", "DEVICE", "PROCEDURE",
                 "BEHAVIORAL", "DIETARY_SUPPLEMENT", "RADIATION",
                 "DIAGNOSTIC_TEST", "OTHER"],
            )
            primary_purpose = st.selectbox(
                "Objectif principal",
                ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE",
                 "SCREENING", "HEALTH_SERVICES_RESEARCH", "BASIC_SCIENCE", "OTHER"],
            )

        with col2:
            st.subheader("Participants et Sites")
            enrollment_count = st.number_input(
                "Nombre de participants", min_value=1, value=150, step=10
            )
            n_locations = st.number_input(
                "Nombre de sites", min_value=0, value=10, step=1
            )
            is_multicenter = st.selectbox(
                "Essai multicentrique", options=[(1, "Oui"), (0, "Non")],
                format_func=lambda x: x[1]
            )[0]
            has_us_site = st.selectbox(
                "Site aux Etats-Unis", options=[(1, "Oui"), (0, "Non")],
                format_func=lambda x: x[1]
            )[0]
            n_collaborators = st.number_input(
                "Nombre de collaborateurs", min_value=0, value=2, step=1
            )

        with col3:
            st.subheader("Protocole")
            n_arms = st.number_input(
                "Nombre de bras", min_value=0, value=2, step=1
            )
            allocation = st.selectbox(
                "Allocation", ["RANDOMIZED", "NON_RANDOMIZED"]
            )
            masking = st.selectbox(
                "Masquage (aveugle)",
                ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
                index=2,
            )
            has_dmc = st.selectbox(
                "Comite de surveillance (DMC)",
                options=[(1, "Oui"), (0, "Non")],
                format_func=lambda x: x[1]
            )[0]
            n_primary_outcomes = st.number_input(
                "Criteres primaires", min_value=0, value=1, step=1
            )
            n_secondary_outcomes = st.number_input(
                "Criteres secondaires", min_value=0, value=3, step=1
            )

        submitted = st.form_submit_button("Predire l'issue de l'essai")

    if submitted:
        payload = {
            "phase": phase,
            "sponsor_type": sponsor_type,
            "enrollment_count": int(enrollment_count),
            "intervention_type": intervention_type,
            "n_arms": int(n_arms),
            "has_dmc": has_dmc,
            "allocation": allocation,
            "masking": masking,
            "primary_purpose": primary_purpose,
            "n_primary_outcomes": int(n_primary_outcomes),
            "n_secondary_outcomes": int(n_secondary_outcomes),
            "n_locations": int(n_locations),
            "is_multicenter": is_multicenter,
            "has_us_site": has_us_site,
            "n_collaborators": int(n_collaborators),
        }
        try:
            with st.spinner("Analyse en cours..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                pred      = data["prediction"]
                proba     = data["probability"]
                conf      = data["confidence"]
                threshold = data["threshold"]

                is_abandon = pred == "abandonne"
                css_class  = "result-abandon" if is_abandon else "result-complete"
                label      = "Risque eleve d'abandon" if is_abandon else "Risque faible d'abandon"
                badge_cls  = f"badge-{conf}"

                conf_labels = {"high": "Confiance elevee", "medium": "Confiance moyenne", "low": "Confiance faible"}
                conf_text = conf_labels.get(conf, conf)

                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <div class="result-title">{label}</div>
                        <div class="result-proba">{proba * 100:.1f} %</div>
                        <div class="result-conf">Probabilite d'abandon</div>
                        <span class="badge {badge_cls}">{conf_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-val">{proba*100:.1f}%</div>'
                        f'<div class="metric-lbl">Probabilite d\'abandon</div></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-val">{threshold}</div>'
                        f'<div class="metric-lbl">Seuil de decision</div></div>',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-val">{conf_text}</div>'
                        f'<div class="metric-lbl">Niveau de confiance</div></div>',
                        unsafe_allow_html=True,
                    )

            elif resp.status_code == 422:
                st.error("Donnees invalides : " + str(resp.json().get("detail")))
            else:
                st.error(f"Erreur API ({resp.status_code}) : {resp.text}")

        except requests.ConnectionError:
            st.error("Impossible de joindre l'API. Verifiez qu'elle est demarree.")
        except Exception as exc:
            st.error(f"Erreur inattendue : {exc}")


# ===========================================================================
# PAGE 2 — PREDICTION EN LOT
# ===========================================================================
elif page == "Prediction en lot":
    st.title("Prediction en lot")
    st.markdown(
        "Uploadez un fichier CSV contenant une ligne par essai clinique. "
        "Les colonnes doivent correspondre aux champs du formulaire de prediction unitaire."
    )

    with st.expander("Format attendu du CSV", expanded=False):
        sample = pd.DataFrame([{
            "phase": "PHASE2", "sponsor_type": "INDUSTRY",
            "enrollment_count": 150, "intervention_type": "DRUG",
            "n_arms": 2, "has_dmc": 1, "allocation": "RANDOMIZED",
            "masking": "DOUBLE", "primary_purpose": "TREATMENT",
            "n_primary_outcomes": 1, "n_secondary_outcomes": 3,
            "n_locations": 10, "is_multicenter": 1,
            "has_us_site": 1, "n_collaborators": 2,
        }])
        st.dataframe(sample, use_container_width=True)
        csv_sample = sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger un fichier exemple",
            csv_sample,
            "exemple_essais.csv",
            "text/csv",
        )

    st.markdown("---")
    uploaded = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            uploaded.seek(0)
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}")
            st.stop()

        st.success(f"Fichier charge — {len(df_preview)} ligne(s), {len(df_preview.columns)} colonne(s)")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("Lancer les predictions"):
            try:
                with st.spinner(f"Prediction sur {len(df_preview)} essai(s)..."):
                    files = {"file": (uploaded.name, uploaded.read(), "text/csv")}
                    resp = requests.post(
                        f"{API_URL}/predict/batch", files=files, timeout=60
                    )

                if resp.status_code == 200:
                    result_df = pd.read_csv(io.StringIO(resp.text))

                    st.markdown("### Resultats")

                    # Metriques rapides
                    n_abandon  = (result_df["prediction"] == "abandonne").sum()
                    n_complete = (result_df["prediction"] == "complete").sum()
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total d'essais", len(result_df))
                    with c2:
                        st.metric("Abandons predits", n_abandon)
                    with c3:
                        st.metric("Completions predites", n_complete)

                    # Tableau colore
                    def color_prediction(val):
                        if val == "abandonne":
                            return "background-color: #4a0000; color: #ef9a9a"
                        return "background-color: #003300; color: #a5d6a7"

                    styled = result_df.style.applymap(
                        color_prediction, subset=["prediction"]
                    )
                    st.dataframe(styled, use_container_width=True)

                    # Telechargement
                    st.download_button(
                        label="Telecharger le CSV enrichi",
                        data=resp.content,
                        file_name=uploaded.name.replace(".csv", "_predictions.csv"),
                        mime="text/csv",
                    )
                else:
                    st.error(f"Erreur API ({resp.status_code}) : {resp.text}")

            except requests.ConnectionError:
                st.error("Impossible de joindre l'API.")
            except Exception as exc:
                st.error(f"Erreur inattendue : {exc}")


# ===========================================================================
# PAGE 3 — INFORMATIONS MODELE
# ===========================================================================
elif page == "Informations modele":
    st.title("Informations sur le modele")
    st.markdown(
        "Cette page affiche les informations techniques du modele deploye, "
        "ses metriques de performance et ses limites d'utilisation."
    )
    st.markdown("---")

    try:
        with st.spinner("Recuperation des informations..."):
            resp = requests.get(f"{API_URL}/model/info", timeout=5)

        if resp.status_code == 200:
            info = resp.json()
            perf = info.get("performance", {})
            cm   = info.get("confusion_matrix", {})

            # --- En-tete modele ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Modele")
                st.info(f"**Type :** {info.get('model_type', 'N/A')}")
                st.info(f"**Strategie de reequilibrage :** {info.get('balancing_strategy', 'N/A')}")
                st.info(f"**Date d'entrainement :** {info.get('training_date', 'N/A')}")
            with col2:
                st.markdown("### Seuils")
                st.info(f"**Seuil deploye (production) :** {info.get('threshold_deployed', 'N/A')}")
                st.info(f"**Seuil optimal Phase 3 :** {info.get('threshold_phase3', 'N/A')}")

            st.markdown("---")

            # --- Metriques de performance ---
            st.markdown("### Metriques de performance (test set, Phase 3)")
            c1, c2, c3, c4, c5 = st.columns(5)
            metrics = [
                ("Recall",     perf.get("recall",    0), c1, "Detection des abandons"),
                ("Precision",  perf.get("precision", 0), c2, "Fiabilite des alertes"),
                ("F1-score",   perf.get("f1",        0), c3, "Equilibre recall/precision"),
                ("Accuracy",   perf.get("accuracy",  0), c4, "Exactitude globale"),
                ("ROC-AUC",    perf.get("roc_auc",   0), c5, "Pouvoir discriminant"),
            ]
            for label, val, col, tooltip in metrics:
                with col:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="metric-val">{val:.2%}</div>'
                        f'<div class="metric-lbl">{label}</div>'
                        f'<div class="metric-lbl" style="font-size:.7rem">{tooltip}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # --- Matrice de confusion ---
            if cm:
                st.markdown("---")
                st.markdown("### Matrice de confusion (test set)")
                cm_df = pd.DataFrame(
                    [
                        [cm.get("true_negatives",  0), cm.get("false_positives", 0)],
                        [cm.get("false_negatives", 0), cm.get("true_positives",  0)],
                    ],
                    index=["Reel : Complete", "Reel : Abandonne"],
                    columns=["Prediction : Complete", "Prediction : Abandonne"],
                )
                st.dataframe(cm_df, use_container_width=False)

            # --- Limites ---
            st.markdown("---")
            st.markdown("### Limites et precautions")
            st.warning(
                "**Ce modele est un outil d'aide a la decision, pas un oracle.** "
                "Il a ete entraine sur des donnees historiques de ClinicalTrials.gov "
                "et peut ne pas generaliser a des essais tres atypiques."
            )
            st.markdown(
                """
                - **Recall = {:.0%}** — environ {:.0%} des essais abandonnes sont manques
                - **Precision = {:.0%}** — certaines alertes sont de fausses alarmes
                - Le seuil de production est oriente **recall** : privilegie la detection au detriment de la precision
                - Les donnees d'entrainement sont anterieures a 2024 — des tendances recentes peuvent ne pas etre capturees
                """.format(
                    perf.get("recall", 0),
                    1 - perf.get("recall", 0),
                    perf.get("precision", 0),
                )
            )

        else:
            st.error(f"Erreur API ({resp.status_code}) : {resp.text}")

    except requests.ConnectionError:
        st.error("Impossible de joindre l'API. Verifiez qu'elle est demarree.")
    except Exception as exc:
        st.error(f"Erreur inattendue : {exc}")