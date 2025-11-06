import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# ------------------- FICHIERS & SAUVEGARDES -------------------
TEAMS_FILE = "teams_form.json"
HISTORIQUE_FILE = "historique_pronos.json"
BACKUP_DIR = "sauvegardes"
os.makedirs(BACKUP_DIR, exist_ok=True)

# Création automatique d'une sauvegarde de teams_form.json
if os.path.exists(TEAMS_FILE):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_DIR, f"teams_form_backup_{date_str}.json")
    shutil.copy(TEAMS_FILE, backup_file)

# ------------------- CHARGEMENT DES DONNÉES -------------------
if os.path.exists(TEAMS_FILE):
    with open(TEAMS_FILE, "r", encoding="utf-8") as f:
        teams_data = json.load(f)
else:
    teams_data = {}

if os.path.exists(HISTORIQUE_FILE):
    with open(HISTORIQUE_FILE, "r", encoding="utf-8") as f:
        historique = json.load(f)
else:
    historique = []

# ------------------- CONFIG STREAMLIT -------------------
st.set_page_config(page_title="Analyseur de matchs complet", layout="wide")
st.title("⚽ Analyseur de matchs – Probabilités & suivi pronostics")

# ------------------- GESTION DES ÉQUIPES -------------------
st.header("🧾 Gestion des équipes")
with st.form("form_teams"):
    team_name = st.text_input("Nom de l'équipe à ajouter ou mettre à jour")
    form_last5 = st.text_input("5 derniers matchs (ex: v,v,n,d,v)")
    goals_scored = st.number_input("Buts marqués", 0, 200, 0)
    goals_against = st.number_input("Buts encaissés", 0, 200, 0)
    submitted_team = st.form_submit_button("💾 Enregistrer l'équipe")

if submitted_team and team_name:
    teams_data[team_name] = {
        "last5": form_last5.lower(),
        "goals_scored": goals_scored,
        "goals_against": goals_against
    }
    with open(TEAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(teams_data, f, indent=2, ensure_ascii=False)
    st.success(f"✅ {team_name} enregistrée avec succès")

# ------------------- AJOUT DE PRONOSTICS -------------------
st.header("📊 Ajouter un pronostic")
if teams_data:
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Équipe Domicile", list(teams_data.keys()))
    with col2:
        away_team = st.selectbox("Équipe Extérieure", [t for t in teams_data.keys() if t != home_team])

    cote_home = st.number_input("Cote Domicile", 1.01, 20.0, 1.5)
    cote_away = st.number_input("Cote Extérieure", 1.01, 20.0, 2.8)

    if st.button("➕ Analyser & Sauvegarder le pronostic"):
        # ----- Calcul de la probabilité -----
        def form_score(seq):
            mapping = {"v":3,"n":1,"d":0}
            vals = [mapping.get(x.strip(),0) for x in seq.split(",") if x.strip() in mapping]
            vals = vals[-5:] if len(vals)>5 else vals
            weights = np.array([5,4,3,2,1])[:len(vals)]
            return np.dot(vals, weights)/(15 if len(vals)==5 else sum(weights))

        form_home = form_score(teams_data[home_team]["last5"])
        form_away = form_score(teams_data[away_team]["last5"])
        p_home_odds = 1 / cote_home
        p_away_odds = 1 / cote_away
        prob_home = (p_home_odds * 0.7 + form_home * 0.3)
        prob_away = (p_away_odds * 0.7 + form_away * 0.3)
        total = prob_home + prob_away
        prob_home /= total
        prob_away /= total

        winner = home_team if prob_home > prob_away else away_team
        prob_victoire = round(max(prob_home, prob_away)*100,2)
        mise = 10

        pronostic = {
            "home_team": home_team,
            "away_team": away_team,
            "cote_home": cote_home,
            "cote_away": cote_away,
            "winner_pred": winner,
            "prob_victoire": prob_victoire,
            "mise": mise,
            "resultat": None,
            "gain": 0
        }

        historique.append(pronostic)
        with open(HISTORIQUE_FILE,"w", encoding="utf-8") as f:
            json.dump(historique,f,indent=2,ensure_ascii=False)
        st.success(f"✅ Pronostic enregistré : victoire de {winner} ({prob_victoire}%)")

else:
    st.warning("⚠️ Ajoute d'abord des équipes avant de pouvoir analyser un match.")

# ------------------- SUIVI DES RESULTATS & STATISTIQUES -------------------
st.header("📅 Suivi des résultats & statistiques")

if historique:
    df = pd.DataFrame(historique)
    st.dataframe(df[["home_team","away_team","winner_pred","prob_victoire","resultat","gain"]], use_container_width=True)

    # ----- Mettre à jour le résultat d’un match -----
    st.subheader("📝 Mettre à jour le résultat d’un match")
    match_index = st.selectbox(
        "Sélectionne un match",
        range(len(historique)),
        format_func=lambda i: f"{historique[i]['home_team']} vs {historique[i]['away_team']}"
    )
    resultat = st.selectbox("Résultat réel", ["home","draw","away"])
    if st.button("✅ Enregistrer le résultat réel"):
        prono = historique[match_index]
        cote = prono["cote_home"] if prono["winner_pred"]==prono["home_team"] else prono["cote_away"]
        if (resultat=="home" and prono["winner_pred"]==prono["home_team"]) or \
           (resultat=="away" and prono["winner_pred"]==prono["away_team"]):
            gain = round(prono["mise"]*cote - prono["mise"],2)
        else:
            gain = -prono["mise"]
        prono["resultat"] = resultat
        prono["gain"] = gain
        with open(HISTORIQUE_FILE,"w", encoding="utf-8") as f:
            json.dump(historique,f,indent=2,ensure_ascii=False)
        st.success(f"Résultat enregistré ✅ (gain : {gain}€)")

    # ----- Statistiques globales -----
    df_valides = df[df["resultat"].notna()]
    if not df_valides.empty:
        total_gain = df_valides["gain"].sum()
        nb_pronos = len(df_valides)
        nb_gagnants = (df_valides["gain"]>0).sum()
        precision = nb_gagnants/nb_pronos*100
        roi = (total_gain/(nb_pronos*10))*100

        st.subheader("📊 Statistiques globales")
        st.metric("🎯 Précision", f"{precision:.2f}%")
        st.metric("💰 ROI", f"{roi:.2f}%")
        st.metric("📈 Gain total", f"{total_gain:.2f}€")

        # Graphique profit cumulé
        df_valides["profit_cumule"] = df_valides["gain"].cumsum()
        fig, ax = plt.subplots()
        ax.plot(df_valides["profit_cumule"], marker='o')
        ax.set_title("Évolution du profit cumulé (€)")
        ax.set_xlabel("Matchs")
        ax.set_ylabel("Profit (€)")
        st.pyplot(fig)

        # Camembert réussite / échec
        fig2, ax2 = plt.subplots()
        nb_perdus = nb_pronos - nb_gagnants
        ax2.pie([nb_gagnants, nb_perdus], labels=["Gagnants","Perdus"], autopct="%1.1f%%", colors=["#4CAF50","#F44336"])
        ax2.set_title("Répartition des pronostics réussis/échoués")
        st.pyplot(fig2)

        # Statistiques par équipe
        st.subheader("🏟️ Statistiques par équipe")
        equipes_stats = {}
        for _, row in df_valides.iterrows():
            for team, won in [(row["home_team"], row["resultat"]=="home"), (row["away_team"], row["resultat"]=="away")]:
                if team not in equipes_stats:
                    equipes_stats[team] = {"joues":0,"gagnes":0,"gain":0}
                equipes_stats[team]["joues"] += 1
                if won:
                    equipes_stats[team]["gagnes"] += 1
                    equipes_stats[team]["gain"] += row["gain"]
                else:
                    equipes_stats[team]["gain"] += row["gain"]
        df_equipes = pd.DataFrame([
            {"Equipe":team,
             "Pronostics joués":v["joues"],
             "Pronostics gagnants":v["gagnes"],
             "Taux de réussite (%)": round(v["gagnes"]/v["joues"]*100,2),
             "Gain total (€)": v["gain"]
            } for team,v in equipes_stats.items()
        ])
        st.dataframe(df_equipes.sort_values("Taux de réussite (%)", ascending=False), use_container_width=True)

    # Export CSV
    st.download_button("📥 Télécharger l’historique complet (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "historique_pronos.csv",
                       "text/csv")

    # Réinitialiser l’historique
    if st.button("🗑️ Réinitialiser l’historique"):
        historique.clear()
        with open(HISTORIQUE_FILE,"w", encoding="utf-8") as f:
            json.dump(historique,f,indent=2,ensure_ascii=False)
        st.warning("Historique réinitialisé.")
else:
    st.info("Aucun pronostic enregistré pour le moment.")
