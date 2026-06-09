# prodadvisor_core.py
# Généré automatiquement — Module de pont pour Streamlit

import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, "/kaggle/working")

CATEGORIES = ['Accessories', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Tops']
INSTRUCTION = (
    "You are a fashion product advisor. "
    "Based on market data, recommend the optimal stock quantity and product type."
)

# ── Chargement des résultats LSTM ──
with open("/kaggle/working/lstm_results.pkl", "rb") as f:
    LSTM_RESULTS = pickle.load(f)

# ── Chargement intelligent du LLM sans passer par Pickle ──
_tokenizer = None
_llm_model = None

def load_llm_lazy():
    """Charge le modèle uniquement lors du premier appel à get_recommendation pour économiser la RAM"""
    global _tokenizer, _llm_model
    if _llm_model is not None:
        return _tokenizer, _llm_model
        
    with open("/kaggle/working/llm_refs.pkl", "rb") as f:
        refs = pickle.load(f)
    
    adapter_path = refs["adapter_path"]
    base_model_name = refs["base_model_name"]
    
    # Configuration et chargement natif
    _tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    _llm_model = PeftModel.from_pretrained(base_model, adapter_path)
    return _tokenizer, _llm_model

def predict_demand(category: str, horizon: int = 1) -> dict:
    if category not in LSTM_RESULTS:
        return {"error": f"Catégorie inconnue : {category}"}
    r = LSTM_RESULTS[category]
    future = r["future"]
    pred_qty = int(future[min(horizon - 1, len(future) - 1)])
    
    trend_pct = ((future[-1] - future[0]) / max(future[0], 1)) * 100
    if trend_pct > 5:
        tendance = f"📈 Hausse +{trend_pct:.1f}%"
    elif trend_pct < -5:
        tendance = f"📉 Baisse {trend_pct:.1f}%"
    else:
        tendance = f"➡️ Stable ({trend_pct:+.1f}%)"
        
    return {
        "predicted_qty": pred_qty,
        "tendance"     : tendance,
        "mae"          : r["mae"],
        "rmse"         : r["rmse"],
        "future"       : [int(v) for v in future],
        "historical"   : [int(v) for v in r["original_data"]],
    }

def get_recommendation(season, category, brand, color, price,
                       rating, markdown, demand, max_new_tokens=120):
    # Initialisation tardive du LLM
    tok, model = load_llm_lazy()
    
    input_text = (
        f"Season: {season}, Category: {category}, Brand: {brand}, "
        f"Color: {color}, Price: {price:.2f}, Rating: {rating:.1f}, "
        f"Markdown: {markdown:.1f}%, Demand: {demand}"
    )
    prompt = (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.3, 
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=1.1,
        )
    decoded = tok.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()
    return response

def map_demand_level(predicted_qty: int) -> str:
    if predicted_qty < 30:
        return "Low Demand"
    elif predicted_qty < 60:
        return "Medium Demand"
    return "High Demand"
