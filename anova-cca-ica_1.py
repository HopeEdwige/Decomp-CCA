import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =========================================================================
# 1. RECONSTITUTION DES DONNÉES BRUTES INDIVIDUELLES
# =========================================================================
# Données basées sur tes fichiers CSV réels pour MUedit (ICA) et sCCA (Labo)
data_muedit = {
    'Condition': (['Normal']*5) + (['PLI']*5) + (['MA']*5) + (['SNR_WGN']*5),
    'MUs_Retrouvees': [47, 47, 47, 47, 47]+ [47, 47, 47, 47, 47] + [43, 42, 47, 45, 42] + [12, 6, 9, 12, 21],
    'Algorithme': ['ICA (MUedit)'] * 20
}

df_muedit = pd.DataFrame(data_muedit)

data_scca = {
    'Condition': (['Normal']*5) + (['PLI']*5) + (['MA']*5) + (['SNR_WGN']*5),
    'MUs_Retrouvees': [14, 14, 14, 14, 14] + [13, 13, 13, 13, 13] + [12, 11, 12, 10, 11] + [4, 6, 6, 13, 11],
    'Algorithme': ['sCCA (Labo)'] * 20
}
df_scca = pd.DataFrame(data_scca)

# Fusion globale
df_global = pd.concat([df_muedit, df_scca], ignore_index=True)

# Calcul du Taux de Réussite (%) par rapport au maximum théorique (79)
UM_theoriques = 79
df_global['Taux_Reussite_Pourcent'] = (df_global['MUs_Retrouvees'] / UM_theoriques) * 100

# =========================================================================
# 2. CALCUL DES P-VALUES VIA L'ANOVA À DEUX FACTEURS
# =========================================================================
model = ols('Taux_Reussite_Pourcent ~ C(Algorithme) * C(Condition)', data=df_global).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Récupération des p-values spécifiques
p_algo = anova_table.loc['C(Algorithme)', 'PR(>F)']
p_cond = anova_table.loc['C(Condition)', 'PR(>F)']
p_inter = anova_table.loc['C(Algorithme):C(Condition)', 'PR(>F)']

# =========================================================================
# 3. CRÉATION DU GRAPHIQUE ET INJECTION DES P-VALUES
# =========================================================================
plt.figure(figsize=(11, 7))

# Couleurs professionnelles (Bleu pour ICA, Rouge/Orange pour sCCA)
palette_colors = {'ICA (MUedit)': '#1f77b4', 'sCCA (Labo)': '#ff7f0e'}

# Dessin des boxplots épaissis avec le paramètre 'hue' pour juxtaposer ICA et sCCA
ax = sns.boxplot(x='Condition', y='Taux_Reussite_Pourcent', hue='Algorithme', 
                 data=df_global, palette=palette_colors, width=0.6, linewidth=1.5)

# Superposition des 5 points réels (répétitions) pour chaque bloc
sns.stripplot(x='Condition', y='Taux_Reussite_Pourcent', hue='Algorithme', 
              data=df_global, color='black', alpha=0.4, size=5, dodge=True, legend=False)

# Configuration esthétique des axes et grilles
plt.title("Comparaison Statistique de la Robustesse de Décomposition\nMUedit (ICA) vs Interface Labo (sCCA)", 
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Condition Clinique / Type d'artefact", fontsize=11, labelpad=10)
plt.ylabel("Taux de Réussite de l'extraction (% des UM)", fontsize=11, labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim([0, 115]) # Espace en haut pour afficher le texte des p-values

# --- AJOUT DU TEXTE DES P-VALUES DE L'ANOVA ---
# Formatage scientifique si la valeur est très petite (ex: 1e-12)
text_p_values = (
    f"Résultats ANOVA à 2 facteurs :\n"
    f"• Effet Algorithme : p = {p_algo:.2e} (Significatif)\n"
    f"• Effet Condition : p = {p_cond:.2e} (Significatif)\n"
    f"• Interaction Algorithme × Condition : p = {p_inter:.2f} (Non-Significatif)"
)

# Encadré texte positionné en haut à gauche du graphique
plt.gca().text(0.02, 0.95, text_p_values, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

# Gestion propre de la légende
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Algorithmes de BSS', loc='upper right', frameon=True)

# Ajustement automatique des marges pour éviter que le texte soit coupé
plt.tight_layout()

# =========================================================================
# 4. EXPORTATION AUTOMATIQUE SOUS FORME D'IMAGE
# =========================================================================
# Sauvegarde en format PNG haute définition (300 DPI), idéal pour l'impression ou l'insertion Word
nom_image_export = 'Comparaison_Graphique_ICA_vs_sCCA.png'
plt.savefig(nom_image_export, dpi=300)

print(f"--- ANALYSE TERMINÉE ---")
print(f"[SUCCÈS] Le graphique comparatif avec les p-values a été exporté sous : '{nom_image_export}'")

# Affichage à l'écran
plt.show()