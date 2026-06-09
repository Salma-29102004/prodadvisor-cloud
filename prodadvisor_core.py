# prodadvisor_core.py
import pickle

# Liste des catégories valides pour l'application
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
    """Charge les résultats du modèle LSTM et renvoie les prédictions associées."""
    try:
        # Lecture locale du dictionnaire de résultats LSTM transféré sur GitHub
        with open("lstm_results.pkl", "rb") as f:
            lstm_data = pickle.load(f)
            
        # Extraction des données associées à la catégorie choisie
        if category in lstm_data:
            cat_data = lstm_data[category]
            historical = cat_data.get("historical", [100, 110, 105, 120])
            future = cat_data.get("future", [130, 140, 150])
            
            # Sélection de la quantité selon l'horizon demandé (Mois +1, +2, +3)
            idx = min(horizon - 1, len(future) - 1)
            predicted_qty = int(future[idx])
            
            # Détermination automatique de la tendance
            tendance = "📈 En hausse" if predicted_qty > historical[-1] else "📉 En baisse"
            
            return {
                "predicted_qty": predicted_qty,
                "tendance": tendance,
                "historical": historical,
                "future": future
            }
        else:
            return {"error": f"Catégorie '{category}' introuvable dans les résultats LSTM."}
            
    except FileNotFoundError:
        return {"error": "Fichier 'lstm_results.pkl' introuvable à la racine du projet GitHub."}
    except Exception as e:
        return {"error": f"Erreur lors du traitement LSTM : {str(e)}"}

def get_recommendation(season, category, brand, color, price, rating, markdown, demand):
    """Génère le rapport stratégique textuel."""
    # Structure de réponse textuelle dynamique simulant les décisions du LLM fine-tuné
    txt = f"### 📋 Rapport Analytique Stratégique — {brand.upper()}\n\n"
    txt += f"**Analyse de l'article :** {category} ({color}) pour la saison *{season}* affiché au prix de {price}€.\n\n"
    
    if "High" in demand:
        txt += f"🔴 **Alerte Stock — Demande Forte :** Le modèle LSTM prévoit une accélération majeure des ventes. " \
               f"Avec une note client de {rating}/5, le produit bénéficie d'une excellente image de marque. " \
               f"**Recommandation :** Augmenter immédiatement les volumes de réapprovisionnement de 25% auprès des fournisseurs. " \
               f"Éviter absolument toute remise (Markdown actuel : {markdown}%) pour maximiser vos marges commerciales."
    elif "Medium" in demand:
        txt += f"🟡 **Optimisation — Demande Stable :** Les volumes de vente attendus restent réguliers. " \
               f"Le positionnement prix de {price}€ est en adéquation avec les standards du marché. " \
               f"**Recommandation :** Maintenir un stock de sécurité constant. Si la rotation ralentit, prévoyez une " \
               f"campagne promotionnelle ciblée ou un ajustement de remise de 10% en fin de saison."
    else:
        txt += f"🔵 **Alerte Surstock — Demande Faible :** Le volume de vente est en baisse critique. " \
               f"**Recommandation :** Stopper immédiatement la production ou l'achat de cette référence. " \
               f"Pour liquider le surstock existant et libérer de l'espace en entrepôt, activez une remise agressive " \
               f"de minimum {max(int(markdown), 30)}% lors des prochaines démarques."
               
    return txt
