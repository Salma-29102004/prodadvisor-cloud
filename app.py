import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prodadvisor_core import predict_demand, get_recommendation, map_demand_level, CATEGORIES

# 1. Configuration de la page
st.set_page_config(
    page_title="ProdAdvisor — Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ ProdAdvisor : Assistant d'Optimisation des Stocks")
st.markdown("Structure prédictive hybride : **LSTM** (Demande) + **LLM Fine-Tuné** (Recommandation)")

# 2. Barre latérale pour les entrées utilisateur (Synchronisée avec vos options Kaggle)
st.sidebar.header("🎯 Paramètres du Produit")

season = st.sidebar.selectbox("Saison", ["Spring", "Summer", "Fall", "Winter"])
category = st.sidebar.selectbox("Catégorie", CATEGORIES)
brand = st.sidebar.selectbox("Marque", ["Zara", "Uniqlo", "H&M", "Gap", "Mango", "Banana Republic", "Ann Taylor", "Forever21"])
color = st.sidebar.selectbox("Couleur", ["Black", "White", "Red", "Blue", "Navy", "Gray", "Beige", "Pink", "Green", "Purple"])

price = st.sidebar.slider("Prix (€)", min_value=10.0, max_value=500.0, value=99.0, step=5.0)
rating = st.sidebar.slider("Note client", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
markdown = st.sidebar.slider("Remise / Markdown (%)", min_value=0.0, max_value=80.0, value=0.0, step=5.0)

horizon = st.sidebar.selectbox("Horizon de prévision LSTM", [1, 2, 3], format_func=lambda x: f"Mois +{x}")

# Flag d'état pour éviter l'exécution automatique à vide au démarrage
if "analyse_lancee" not in st.session_state:
    st.session_state.analyse_lancee = False

if st.sidebar.button("📊 Générer l'Analyse & Recommandation", type="primary"):
    st.session_state.analyse_lancee = True

# 3. Logique principale d'affichage
if st.session_state.analyse_lancee:
    with st.spinner("Calcul des prévisions de vente (LSTM) et génération des conseils (LLM)..."):
        # Appel du bloc LSTM sécurisé via le fichier core
        lstm_out = predict_demand(category, horizon=horizon)
        
        if "error" in lstm_out:
            st.error(lstm_out["error"])
        else:
            # Récupération sécurisée des données LSTM pour la catégorie choisie
            qty = lstm_out["predicted_qty"]
            tendance = lstm_out["tendance"]
            demand_level = map_demand_level(qty)
            
            # Affichage des indicateurs clés (KPIs)
            st.subheader("📈 1. Analyse Prédictive du Marché (Modèle LSTM)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label=f"Quantité prédite (Mois +{horizon})", value=f"{qty} unités")
            with col2:
                st.metric(label="Tendance du marché", value=tendance)
            with col3:
                st.metric(label="Niveau de Demande estimé", value=demand_level)
                
            # Tableau récapitulatif global (Lignes 218-224 corrigées et sécurisées)
            st.markdown("### 📋 Vue d'ensemble de toutes les catégories")
            rows = []
            for cat in CATEGORIES:
                r = predict_demand(cat)
                # Sécurisation stricte : on vérifie que "future" et "tendance" existent avant de lire
                if isinstance(r, dict) and "future" in r and "tendance" in r:
                    rows.append({
                        "Catégorie" : cat,
                        "Mois +1"   : r["future"][0] if len(r["future"]) > 0 else 0,
                        "Mois +2"   : r["future"][1] if len(r["future"]) > 1 else 0,
                        "Mois +3"   : r["future"][2] if len(r["future"]) > 2 else 0,
                        "Tendance"  : r["tendance"]
                    })
                else:
                    rows.append({
                        "Catégorie" : cat,
                        "Mois +1": 0, "Mois +2": 0, "Mois +3": 0, "Tendance": "Indisponible"
                    })
            
            df_summary = pd.DataFrame(rows)
            st.dataframe(df_summary, use_container_width=True)
            
            # Graphique d'historique et prévisions Plotly pour la catégorie sélectionnée
            st.markdown(f"**Évolution de la demande historique et future pour {category} :**")
            chart_data = pd.DataFrame({
                "Mois": [f"H-{i}" for i in range(len(lstm_out["historical"]), 0, -1)] + ["Mois +1", "Mois +2", "Mois +3"],
                "Volume": lstm_out["historical"] + lstm_out["future"][:3]
            })
            fig = px.line(chart_data, x="Mois", y="Volume", markers=True, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Appel du bloc LLM pour le rapport textuel
            st.subheader("🤖 2. Rapport Stratégique (LLM Fine-Tuné)")
            
            recommendation_text = get_recommendation(
                season=season,
                category=category,
                brand=brand,
                color=color,
                price=price,
                rating=rating,
                markdown=markdown,
                demand=demand_level
            )
            
            st.info(recommendation_text)
            st.success("Analyse terminée avec succès !")
else:
    # Message d'accueil propre avant le premier clic sur le bouton
    st.info("👉 Renseignez les paramètres de votre produit dans la barre latérale gauche, puis cliquez sur 'Générer l'Analyse & Recommandation' pour lancer l'analyse.")
