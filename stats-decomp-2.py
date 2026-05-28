import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. TES VRAIES DONNÉES EXACTES
# ==============================================================================
# --- Scores MUedit (ICA) ---
ica_normal = [6, 6, 6, 6, 6]
ica_pli    = [6, 6, 6, 6, 6]
ica_ma     = [9, 10, 9, 11, 7] 
ica_snr    = [0, 0, 0, 1, 1]

# --- Scores Ton Logiciel (CCA) ---
cca_normal = [28, 28, 28, 28, 28]
cca_pli    = [28, 28, 28, 28, 28]
cca_ma     = [25, 18, 24, 23, 17]
cca_snr    = [47, 46, 45, 38, 31]

# --- Scores Ton Logiciel (sCCA) ---
scca_normal = [26, 26, 26, 26, 26]
scca_pli    = [30, 30, 30, 30, 30]
scca_ma     = [23, 21, 25, 19, 22]
scca_snr    = [46, 47, 42, 40, 30]

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
# Fusion de toutes les listes
tous_les_scores = (ica_normal + ica_pli + ica_ma + ica_snr + 
                   cca_normal + cca_pli + cca_ma + cca_snr +
                   scca_normal + scca_pli + scca_ma + scca_snr)

conditions = (['Normal']*5 + ['PLI']*5 + ['MA']*5 + ['SNR_WGN']*5) * 3
algorithmes = (['ICA (MUedit)'] * 20) + (['CCA'] * 20) + (['sCCA'] * 20)

df = pd.DataFrame({
    'Algorithme': algorithmes,
    'Condition': conditions,
    'MUs_Trouvees': tous_les_scores
})

# ==============================================================================
# 3. CRÉATION DU GRAPHIQUE (Boxplot groupé par 3)
# ==============================================================================
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")

# Palette de couleurs pour les 3 méthodes
couleurs = {"ICA (MUedit)": "#4C72B0", "CCA": "#DD8452", "sCCA": "#55A868"}

# Tracé du Boxplot
ax = sns.boxplot(x="Condition", y="MUs_Trouvees", hue="Algorithme", data=df, 
                 palette=couleurs, showmeans=True, 
                 meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize": 7})

# Tracé des points individuels
sns.stripplot(x="Condition", y="MUs_Trouvees", hue="Algorithme", data=df, 
              palette=couleurs, alpha=0.8, dodge=True, ax=ax, legend=False,
              linewidth=1, edgecolor='black', size=5)

# ==============================================================================
# 4. FINITIONS VISUELLES
# ==============================================================================
plt.title('Comparaison de l\'extraction : ICA vs CCA vs sCCA sur modèle Fair Benchmark (2048 Hz)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Nombre d\'Unités Motrices Extraites', fontsize=12, fontweight='bold')
plt.xlabel('Condition de Simulation (Bruit)', fontsize=12, fontweight='bold')

plt.legend(title='Algorithme', loc='upper right')
plt.tight_layout()

# Affichage
plt.show()

# ==============================================================================
# 5. EXPORT DU RAPPORT DE SYNTHÈSE (Console)
# ==============================================================================
print("\n" + "="*50)
print("=== RAPPORT DES MOYENNES (TES DONNÉES) ===")
print("="*50)
resume_stats = df.groupby(['Condition', 'Algorithme'])['MUs_Trouvees'].agg(['mean', 'std']).round(2)
resume_stats.columns = ['Moyenne_UM', 'Ecart_Type']
print(resume_stats.to_string())
print("="*50 + "\n")