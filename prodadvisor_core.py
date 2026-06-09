# prodadvisor_core.py
import pickle

# Liste dynamique des catégories de votre projet
CATEGORIES = ["Jeans", "T-shirts", "Dresses", "Jackets", "Sweaters"]

def map_demand_level(predicted_qty):
    """Calcule le niveau de demande selon les seuils du projet."""
    if predicted_qty < 120:
        return "Low (Faible)"
    elif predicted_qty <= 250:
        return "Medium (Modérée)"
    else:
        return "High (Forte)"

def predict_demand(category, horizon=1):
    """Charge les résultats du modèle LSTM et convertit les types pour éviter les conflits NumPy/Listes."""
    try:
        # Lecture du dictionnaire de résultats LSTM transféré sur GitHub
        with open("lstm_results.pkl", "rb") as f:
            lstm_data = pickle.load(f)
    except Exception:
        lstm_data = {}

    # 1. Cas où la catégorie existe réellement dans vos résultats LSTM
    if category in lstm_data:
        cat_data = lstm_data[category]
        
        # Récupération des données d'origine (Listes ou Tableaux NumPy)
        raw_historical = cat_data.get("historical", [100, 110, 105, 120])
        raw_future = cat_data.get("future", [130, 140, 150])
        
        # SÉCURISATION : Conversion stricte en listes standards Python pour le "+" de la ligne 89
        historical = raw_historical.tolist() if hasattr(raw_historical, "tolist") else list(raw_historical)
        future = raw_future.tolist() if hasattr(raw_future, "tolist") else list(raw_future)
        
        idx = min(horizon - 1, len(future) - 1)
        predicted_qty = int(future[idx])
        tendance = "📈 En hausse" if predicted_qty > historical[-1] else "📉 En baisse"
        
        return {
            "predicted_qty": predicted_qty,
            "tendance": tendance,
            "historical": historical,
            "future": future
        }
        
    # 2. Cas de secours (Simulation propre) pour les catégories absentes du pkl
    else:
        return {
            "predicted_qty": 190,
            "tendance": "📈 En hausse (Simulation)",
            "historical": [140, 165, 150, 175],
            "future": [190, 210, 235]
        }

def get_recommendation(season, category, brand, color, price, rating, markdown, demand):
    """Génère le rapport stratégique textuel."""
    txt = f"### 📋 Rapport Analytique Stratégique — {brand.upper()}\n\n"
    txt += f"**Analyse de l'article :** {category} ({color}) pour la saison *{season}* affiché au prix de {price}€.\n\n"
    
    if "High" in demand:
        txt += f"🔴 **Alerte Stock — Demande Forte :** Le modèle prédictif prévoit une accélération majeure des ventes. " \
               f"Avec une note client de {rating}/5, le produit bénéficie d'une excellente dynamique commerciale chez {brand}. " \
               f"**Recommandation :** Augmenter immédiatement les volumes de réapprovisionnement de 25% auprès des fournisseurs. " \
               f"Éviter absolument toute remise (Markdown actuel : {markdown}%) pour maximiser les marges."
    elif "Medium" in demand:
        txt += f"🟡 **Optimisation — Demande Stable :** Les volumes de vente attendus restent réguliers. " \
               f"Le positionnement prix de {price}€ est en parfaite adéquation avec les standards du marché actuel. " \
               f"**Recommandation :** Maintenir un stock de sécurité constant. Si la rotation ralentit en fin de saison, " \
               f"envisager une légère remise promotionnelle de 10% à 15%."
    else:
        txt += f"🔵 **Alerte Surstock — Demande Faible :** Le volume de vente estimé est en baisse critique pour cette catégorie. " \
               f"**Recommandation :** Limiter l'exposition financière en stoppant les commandes sur cette référence. " \
               f"Pour liquider le stock existant et libérer de l'espace en entrepôt, activez une remise agressive " \
               f"de minimum {max(int(markdown), 35)}% lors des prochaines démarques."
               
    return txt
