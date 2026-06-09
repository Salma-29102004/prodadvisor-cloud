
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

# ── Configuration page ──────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ProdAdvisor",
    page_icon  = "🛍️",
    layout     = "wide",
)

# ── Chargement des fonctions depuis le notebook (via variables globales) ─────
# Les fonctions predict_demand() et get_recommendation() sont
# injectées dans le namespace de l'app via st.session_state
# (voir cellule step4-launch qui lance streamlit avec --global)
# Alternativement elles sont importées depuis prodadvisor_core.py
try:
    from prodadvisor_core import predict_demand, get_recommendation, map_demand_level, CATEGORIES
except ImportError:
    st.error("❌ Impossible d'importer prodadvisor_core.py — lancer la cellule 4.4 d'abord")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛍️ ProdAdvisor")
st.caption("Recommandation de stock basée sur prédiction LSTM + LLM fine-tuné")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Paramètres produit")

    season = st.selectbox(
        "Saison",
        options=["Spring", "Summer", "Fall", "Winter"],
        index=2,
    )
    category = st.selectbox(
        "Catégorie",
        options=CATEGORIES,
    )
    brand = st.selectbox(
        "Marque",
        options=["Zara", "Uniqlo", "H&M", "Gap", "Mango",
                 "Banana Republic", "Ann Taylor", "Forever21"],
    )
    color = st.selectbox(
        "Couleur",
        options=["Black", "White", "Red", "Blue", "Navy", "Gray",
                 "Beige", "Pink", "Green", "Purple"],
    )
    price = st.slider(
        "Prix (€)",
        min_value=10.0, max_value=500.0,
        value=99.0, step=5.0,
    )
    rating = st.slider(
        "Note client",
        min_value=1.0, max_value=5.0,
        value=3.5, step=0.1,
    )
    markdown = st.slider(
        "Réduction markdown (%)",
        min_value=0.0, max_value=80.0,
        value=10.0, step=5.0,
    )
    horizon = st.radio(
        "Horizon de prédiction",
        options=[1, 2, 3],
        format_func=lambda x: f"Mois +{x}",
        horizontal=True,
    )

    run_btn = st.button("🚀 Analyser", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:

    # ── 1. LSTM predict ──────────────────────────────────────────────────────
    with st.spinner("⏳ Prédiction LSTM en cours..."):
        lstm_result = predict_demand(category, horizon)

    pred_qty = lstm_result["predicted_qty"]
    tendance = lstm_result["tendance"]
    demand   = map_demand_level(pred_qty)

    # ── 2. LLM recommend ─────────────────────────────────────────────────────
    with st.spinner("🤖 Génération de la recommandation (LLM)..."):
        recommendation = get_recommendation(
            season   = season,
            category = category,
            brand    = brand,
            color    = color,
            price    = price,
            rating   = rating,
            markdown = markdown,
            demand   = demand,
        )

    # ── 3. Affichage métriques clés ──────────────────────────────────────────
    st.subheader("📊 Résultats")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label  = f"Quantité recommandée (M+{horizon})",
        value  = f"{pred_qty} unités",
    )
    col2.metric(
        label = "Tendance (3 mois)",
        value = tendance,
    )
    col3.metric(
        label = "Niveau de demande",
        value = demand,
    )
    col4.metric(
        label = "Précision LSTM (MAE)",
        value = f"± {lstm_result['mae']} unités",
    )

    st.divider()

    # ── 4. Recommandation LLM ────────────────────────────────────────────────
    col_rec, col_chart = st.columns([1, 1])

    with col_rec:
        st.subheader("💬 Recommandation")
        st.info(recommendation)

        st.caption(f"**Input transmis au LLM :**")
        st.code(
            f"Season: {season}, Category: {category}, Brand: {brand}, "
            f"Color: {color}, Price: {price:.2f}, Rating: {rating:.1f}, "
            f"Markdown: {markdown:.1f}%, Demand: {demand}",
            language="text",
        )

    # ── 5. Graphique tendance ────────────────────────────────────────────────
    with col_chart:
        st.subheader("📈 Évolution des stocks")

        hist   = lstm_result["historical"]
        future = lstm_result["future"]
        n      = len(hist)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = list(range(n)),
            y    = hist,
            mode = "lines+markers",
            name = "Historique",
            line = dict(color="#4A90D9", width=2),
        ))
        fig.add_trace(go.Scatter(
            x    = list(range(n - 1, n + 3)),
            y    = [hist[-1]] + future,
            mode = "lines+markers",
            name = "Prédiction LSTM",
            line = dict(color="#E24B4A", width=2, dash="dash"),
            marker = dict(symbol="square", size=8),
        ))
        # Marquer le point sélectionné
        fig.add_trace(go.Scatter(
            x    = [n + horizon - 1],
            y    = [pred_qty],
            mode = "markers",
            name = f"M+{horizon} sélectionné",
            marker = dict(color="#EF9F27", size=14, symbol="star"),
        ))
        fig.add_vrect(
            x0        = n - 0.5,
            x1        = n + 2.5,
            fillcolor = "rgba(226,75,74,0.07)",
            line_width= 0,
        )
        fig.update_layout(
            xaxis_title  = "Mois",
            yaxis_title  = "Quantité",
            legend       = dict(orientation="h", yanchor="bottom", y=1.02),
            margin       = dict(l=0, r=0, t=30, b=0),
            height       = 320,
            plot_bgcolor = "rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 6. Tableau toutes catégories ─────────────────────────────────────────
    st.divider()
    st.subheader("📋 Vue d'ensemble — toutes catégories")

    rows = []
    for cat in CATEGORIES:
        r = predict_demand(cat, horizon)
        rows.append({
            "Catégorie"        : cat,
            f"Mois +{horizon}" : r["predicted_qty"],
            "Tendance"         : r["tendance"],
            "MAE (unités)"     : r["mae"],
            "RMSE (unités)"    : r["rmse"],
        })

    df_overview = pd.DataFrame(rows)
    st.dataframe(df_overview, use_container_width=True, hide_index=True)

else:
    st.info("👈 Renseigne les paramètres dans la sidebar et clique sur **Analyser**")

    # Preview du tableau vide
    st.subheader("📋 Aperçu — prédictions LSTM (3 mois)")
    rows = []
    for cat in CATEGORIES:
        r = predict_demand(cat)
        rows.append({
            "Catégorie" : cat,
            "Mois +1"   : r["future"][0],
            "Mois +2"   : r["future"][1],
            "Mois +3"   : r["future"][2],
            "Tendance"  : r["tendance"],
            "MAE"       : r["mae"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
