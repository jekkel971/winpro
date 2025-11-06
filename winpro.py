import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Analyseur de matchs avancé", layout="wide")
st.title("⚽ Analyseur de matchs – Probabilités réalistes avec cotes")

# ---------------- FICHIERS ET SAUVEGARDES ----------------
FORM_FILE = "teams_form.json"
BACKUP_DIR = "sauvegardes"
HISTORY_FILE = "pronostics_history.json"

os.makedirs(BACKUP_DIR, exist_ok=True)

# Sauvegarde automatique des équipes au démarrage
if os.path.exists(FORM_FILE):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_DIR, f"teams_form_backup_{date_str}.json")
    shutil.copy(FORM_FILE, backup_file)

# Charger équipes existantes
if os.path.exists(FORM_FILE):
    with open(FORM_FILE,"r") as f:
        teams_form = json.load(f)
else:
    teams_form = {}

# Charger historique pronostics
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE,"r") as f:
        pronostics_history = json.load(f)
else:
    pronostics_history = []

# ---------------- DATAFRAME MATCHES ----------------
if "matches_df" not in st.session_state:
    st.session_state.matches_df = pd.DataFrame(columns=[
        "home_team","away_team","cote_home","cote_away",
        "home_wins","home_draws","home_losses","home_goals_scored","home_goals_against","home_last5",
        "away_wins","away_draws","away_losses","away_goals_scored","away_goals_against","away_last5",
        "Winner"
    ])

# ---------------- SÉLECTION RAPIDE DES ÉQUIPES ----------------
st.subheader("Sélection rapide des équipes existantes")
saved_teams = sorted(teams_form.keys())
if saved_teams:
    selected_team = st.selectbox("Choisir une équipe existante", saved_teams)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"🏠 Mettre {selected_team} en domicile"):
            st.session_state.home_team = selected_team
    with col2:
        if st.button(f"🛫 Mettre {selected_team} en extérieur"):
            st.session_state.away_team = selected_team

home_team_default = st.session_state.get("home_team", "")
away_team_default = st.session_state.get("away_team", "")

# ---------------- FORMULAIRE ----------------
with st.form("match_form", clear_on_submit=True):
    st.subheader("Équipes et cotes")
    home_team = st.text_input("Équipe domicile", value=home_team_default)
    away_team = st.text_input("Équipe extérieure", value=away_team_default)
    cote_home = st.number_input("Cote domicile", 1.01, 10.0, 1.5)
    cote_away = st.number_input("Cote extérieure", 1.01, 10.0, 1.5)

    st.subheader("Historique domicile")
    home_wins = st.number_input("Victoires domicile", 0, 50, 0)
    home_draws = st.number_input("Nuls domicile", 0, 50, 0)
    home_losses = st.number_input("Défaites domicile", 0, 50, 0)
    home_goals_scored = st.number_input("Buts marqués domicile", 0, 200, 0)
    home_goals_against = st.number_input("Buts encaissés domicile", 0, 200, 0)
    default_home_last5 = teams_form.get(home_team,"v,v,n,d,d") if home_team else "v,v,n,d,d"
    home_last5 = st.text_input("5 derniers matchs domicile (v,n,d)", value=default_home_last5)

    st.subheader("Historique extérieur")
    away_wins = st.number_input("Victoires extérieur", 0, 50, 0)
    away_draws = st.number_input("Nuls extérieur", 0, 50, 0)
    away_losses = st.number_input("Défaites extérieur", 0, 50, 0)
    away_goals_scored = st.number_input("Buts marqués extérieur", 0, 200, 0)
    away_goals_against = st.number_input("Buts encaissés extérieur", 0, 200, 0)
    default_away_last5 = teams_form.get(away_team,"v,v,n,d,d") if away_team else "v,v,n,d,d"
    away_last5 = st.text_input("5 derniers matchs extérieur (v,n,d)", value=default_away_last5)

    submitted = st.form_submit_button("➕ Ajouter le match")

# ---------------- AJOUT DES DONNÉES ----------------
if submitted and home_team and away_team:
    teams_form[home_team] = home_last5.lower()
    teams_form[away_team] = away_last5.lower()
    with open(FORM_FILE,"w") as f:
        json.dump(teams_form,f)

    # Ajout au DataFrame
    st.session_state.matches_df = pd.concat([
        st.session_state.matches_df,
        pd.DataFrame([{
            "home_team": home_team,"away_team": away_team,
            "cote_home": cote_home,"cote_away": cote_away,
            "home_wins": home_wins,"home_draws": home_draws,"home_losses": home_losses,
            "home_goals_scored": home_goals_scored,"home_goals_against": home_goals_against,"home_last5": home_last5.lower(),
            "away_wins": away_wins,"away_draws": away_draws,"away_losses": away_losses,
            "away_goals_scored": away_goals_scored,"away_goals_against": away_goals_against,"away_last5": away_last5.lower()
        }])
    ], ignore_index=True)
    st.success(f"✅ Match ajouté : {home_team} vs {away_team}")

# ---------------- FONCTIONS D’ANALYSE ----------------
def calculate_form_score(sequence):
    mapping = {"v":3,"n":1,"d":0}
    seq = [mapping.get(x.strip(),0) for x in sequence.split(",")]
    if len(seq)<5: seq+=[0]*(5-len(seq))
    weights=np.array([5,4,3,2,1])
    return np.dot(seq,weights)/15

def calculate_prob(home_last5, away_last5, home_goals, home_against, away_goals, away_against, cote_home, cote_away):
    home_form = calculate_form_score(home_last5)
    away_form = calculate_form_score(away_last5)

    home_attack = home_goals / max(home_goals+home_against,1)
    away_attack = away_goals / max(away_goals+away_against,1)

    home_score = 0.5*home_form + 0.25*home_attack + 0.25*(1-away_attack)
    away_score = 0.5*away_form + 0.25*away_attack + 0.25*(1-home_attack)

    prob_home_cote = 1 / cote_home
    prob_away_cote = 1 / cote_away

    prob_home = 0.7*home_score + 0.3*prob_home_cote
    prob_away = 0.7*away_score + 0.3*prob_away_cote

    total = prob_home + prob_away
    prob_home /= total
    prob_away /= total
    return prob_home, prob_away

def analyze(df):
    df = df.copy()
    results=[]
    for _,row in df.iterrows():
        prob_home, prob_away = calculate_prob(
            row["home_last5"], row["away_last5"],
            row["home_goals_scored"], row["home_goals_against"],
            row["away_goals_scored"], row["away_goals_against"],
            row["cote_home"], row["cote_away"]
        )
        winner = row["home_team"] if prob_home>prob_away else row["away_team"]
        results.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "Winner": winner,
            "Probabilité victoire": round(max(prob_home,prob_away)*100,2),
            "Score Sécurité": round(abs(prob_home-prob_away)*100,1)
        })
    return pd.DataFrame(results)

def update_form_after_match(df_analysis):
    for idx,row in df_analysis.iterrows():
        winner = row["Winner"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_seq = teams_form.get(home_team,"v,v,n,d,d").split(",")[:4]
        away_seq = teams_form.get(away_team,"v,v,n,d,d").split(",")[:4]

        if winner==home_team:
            home_seq=["v"]+home_seq
            away_seq=["d"]+away_seq
        elif winner==away_team:
            home_seq=["d"]+home_seq
            away_seq=["v"]+away_seq
        else:
            home_seq=["n"]+home_seq
            away_seq=["n"]+away_seq

        teams_form[home_team]=",".join(home_seq)
        teams_form[away_team]=",".join(away_seq)

    with open(FORM_FILE,"w") as f:
        json.dump(teams_form,f)

# ---------------- AFFICHAGE ANALYSE ----------------
if len(st.session_state.matches_df)>0:
    st.subheader("📊 Analyse des matchs")
    df_analysis = analyze(st.session_state.matches_df)
    df_analysis = df_analysis.sort_values(by="Score Sécurité",ascending=False)
    st.dataframe(df_analysis[["home_team","away_team","Winner","Probabilité victoire","Score Sécurité"]],use_container_width=True)

    st.subheader("💰 Recommandation de mise (Kelly simplifié)")
    budget_total = st.number_input("Budget total (€)",1,10000,100,step=10)
    df_analysis["cote_home"] = st.session_state.matches_df["cote_home"]
    df_analysis["cote_away"] = st.session_state.matches_df["cote_away"]
    mises=[]
    for i,row in df_analysis.iterrows():
        cote = row["cote_home"] if row["Winner"]==row["home_team"] else row["cote_away"]
        p = row["Probabilité victoire"]/100
        b = cote-1
        q = 1-p
        f_star = max((b*p-q)/b,0)
        mises.append(round(f_star*budget_total,2))
    df_analysis["Mise conseillée (€)"]=mises
    st.dataframe(df_analysis[["home_team","away_team","Winner","Probabilité victoire","Score Sécurité","Mise conseillée (€)"]],use_container_width=True)

    update_form_after_match(df_analysis)
    st.success("✅ Formes mises à jour automatiquement")

    st.download_button("📥 Télécharger résultats (CSV)", df_analysis.to_csv(index=False).encode("utf-8"), "analyse_matchs.csv","text/csv")
else:
    st.info("Ajoute au moins un match pour commencer l’analyse ⚙️")

# ---------------- ÉVALUATION DES PRONOSTICS ----------------
st.subheader("🏁 Évaluer les pronostics passés")

if len(st.session_state.matches_df) > 0:
    evaluation = []
    for i, row in st.session_state.matches_df.iterrows():
        st.markdown(f"**{row['home_team']} vs {row['away_team']}**")
        real_winner = st.radio(
            "Résultat réel :",
            options=[row['home_team'], row['away_team'], "Match nul"],
            key=f"real_{i}"
        )
        evaluation.append({
            "home_team": row['home_team'],
            "away_team": row['away_team'],
            "predicted_winner": row.get("Winner",""),
            "real_winner": real_winner
        })

    if st.button("✅ Calculer les résultats"):
        correct = 0
        for match in evaluation:
            if match['predicted_winner'] == match['real_winner']:
                correct += 1
        total = len(evaluation)
        st.success(f"Tu as eu {correct} pronostics corrects sur {total} ({round(correct/total*100,2)}%)")

        # Sauvegarde historique
        pronostics_history.extend(evaluation)
        with open(HISTORY_FILE,"w") as f:
            json.dump(pronostics_history,f)
        st.info(f"Historique des pronostics mis à jour ({len(pronostics_history)} entrées)")

# ---------------- FIN ----------------
