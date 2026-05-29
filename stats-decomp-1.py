import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =========================================================================
# 1. SAISIE MANUELLE DE TES RÉSULTATS DE DECOMPOSITION (NOMBRE D'UM RETROUVÉES)
# =========================================================================
# Rentre ici les 5 valeurs du nombre d'unités motrices (UM) lues dans Decomp
MUs_Normal = [6, 6, 6, 6, 6]  # Groupe Contrôle (Signal pur de SYNCHRO)
MUs_PLI    = [6, 6, 6, 6, 6]  # Interférence ligne électrique (50Hz + harmoniques)
MUs_MA     = [9, 10, 9, 11, 7]  # Artefact de mouvement involontaire (dérive + secousse)
MUs_SNR    = [0, 0, 0, 1, 1]  # Dégradation par bruit blanc (SNR : 3, 5, 11, 15, 20 dB)

# =========================================================================
# 2. STRUCTURATION ET PREPARATION DU TABLEAU DE DONNÉES (DATAFRAME)
# =========================================================================
# On fusionne toutes les listes de valeurs en un seul grand vecteur de 20 éléments
valeurs = MUs_Normal + MUs_PLI + MUs_MA + MUs_SNR

# On crée le vecteur des étiquettes (facteurs) correspondant à chaque mesure
conditions = (['Normal'] * 5) + (['PLI'] * 5) + (['MA'] * 5) + (['SNR_WGN'] * 5)

# Création du DataFrame Pandas (structure idéale pour les modèles statistiques)
df = pd.DataFrame({'Condition': conditions, 'MUs_Retrouvees': valeurs})

# =========================================================================
# 3. CALCUL, AFFICHAGE ET ENREGISTREMENT DES STATISTIQUES DESCRIPTIVES
# =========================================================================
print("--- MÉTRIQUES DESCRIPTIVES ---")
# groupby regroupe par condition, agg calcule la moyenne ('mean') et l'écart-type ('std')
stats_descriptives = df.groupby('Condition')['MUs_Retrouvees'].agg(['mean', 'std'])
# On renomme les colonnes pour que ce soit plus propre
stats_descriptives.columns = ['Moyenne_UM', 'Ecart_Type_UM']
print(stats_descriptives)
print("\n" + "="*40 + "\n")

# --- SAUVEGARDE DES CALCULS ---
# Option A : Enregistrement sous forme de fichier CSV (ouvrable directement dans Excel)
stats_descriptives.to_csv('Resultats_Moyennes_Decomp.csv')

# Option B : Enregistrement dans un fichier texte d'analyse complet
with open("Rapport_Stats_Synthese.txt", "w", encoding="utf-8") as f:
    f.write("=== RAPPORT DE SYNTHÈSE DES DECOMPOSITIONS Decomp ===\n\n")
    f.write("--- STATISTIQUES DESCRIPTIVES ---\n")
    f.write(stats_descriptives.to_string())
    f.write("\n\n" + "="*40 + "\n\n")

# =========================================================================
# 4. LE TEST DE L'ANOVA (Analyse de Variance à 1 facteur)
# =========================================================================
# ols() définit le modèle linéaire : la variable à expliquer dépend du facteur 'Condition'
# .fit() calcule les moindres carrés du modèle
model = ols('MUs_Retrouvees ~ C(Condition)', data=df).fit()

# anova_lm calcule la variance inter-groupe et intra-groupe et génère la p-value
anova_table = sm.stats.anova_lm(model, typ=2)
print("--- TABLE ANOVA ---")
print(anova_table)
print("\n" + "="*40 + "\n")

# Sauvegarde de la table ANOVA dans le fichier texte
with open("Rapport_Stats_Synthese.txt", "a", encoding="utf-8") as f:
    f.write("--- TABLE DE L'ANOVA ---\n")
    f.write(anova_table.to_string())
    f.write("\n\n" + "="*40 + "\n\n")

# =========================================================================
# 5. TEST POST-HOC (TUKEY HSD - Comparaisons par paires)
# =========================================================================
# L'ANOVA dit s'il y a une différence globale. Le test de Tukey (HSD) compare 
# les conditions 2 à 2 pour identifier précisément où se trouvent les écarts significatifs.
print("--- TEST POST-HOC DE TUKEY ---")
tukey = pairwise_tukeyhsd(endog=df['MUs_Retrouvees'], groups=df['Condition'], alpha=0.05)
print(tukey)

# Sauvegarde du résultat de Tukey dans le fichier texte
with open("Rapport_Stats_Synthese.txt", "a", encoding="utf-8") as f:
    f.write("--- TEST POST-HOC DE TUKEY (alpha=0.05) ---\n")
    f.write(str(tukey))
    f.write("\n")

print("\n[INFO] Les calculs et résultats ont été enregistrés dans 'Resultats_Moyennes_Decomp.csv' et 'Rapport_Stats_Synthese.txt' !")

# =========================================================================
# 6. REPRÉSENTATION GRAPHIQUE (Boxplot + Points individuels)
# =========================================================================
plt.figure(figsize=(8, 6))

# Dessine les boîtes à moustaches (médiane, quartiles, minimum, maximum)
sns.boxplot(x='Condition', y='MUs_Retrouvees', data=df, palette='Set2', width=0.5)

# Stripplot superpose les 5 points réels de tes répétitions par-dessus les boîtes
sns.stripplot(x='Condition', y='MUs_Retrouvees', data=df, color='black', alpha=0.6, size=6)

# Paramétrage des titres et axes
plt.title("Impact des artefacts sur l'extraction des UM (Decomp - CCA)", fontsize=12, fontweight='bold')
plt.xlabel("Condition / Type d'artefact appliqué", fontsize=10)
plt.ylabel("Nombre d'Unités Motrices extraites (UM)", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Ajoute une grille horizontale discrète

# Affichage du graphique à l'écran
plt.show()