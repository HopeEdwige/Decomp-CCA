import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==============================================================================
# 1. RENSEIGNE TES DONNÉES ICI (Les 5 répétitions pour chaque condition)
# ==============================================================================
# --- Scores de MUedit (ICA) ---
ica_normal = [47, 47, 47, 47, 47]
ica_pli    = [47, 47, 47, 47, 47]
ica_ma     = [43, 42, 47, 45, 42] 
ica_snr    = [12, 6, 9, 12, 21] 

# --- Scores du Logiciel (CCA) ---
cca_normal = [14, 14, 14, 14, 14]
cca_pli    = [13, 13, 13, 13, 13]
cca_ma     = [12, 11, 12, 10, 11]
cca_snr    = [4, 6, 6, 13, 11]


# ==============================================================================
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
data = {
    'Algorithme': ['ICA (MUedit)']*20 + ['CCA (Decomp-CCA)']*20,
    'Condition': ['Normal']*5 + ['PLI']*5 + ['MA']*5 + ['SNR_WGN']*5 + 
                 ['Normal']*5 + ['PLI']*5 + ['MA']*5 + ['SNR_WGN']*5,
    'MUs_Trouvees': ica_normal + ica_pli + ica_ma + ica_snr + 
                    cca_normal + cca_pli + cca_ma + cca_snr
}
df = pd.DataFrame(data)

# ==============================================================================
# 3. FONCTION POUR DESSINER LES BARRES DE SIGNIFICATIVITÉ
# ==============================================================================
def draw_p_value_bar(ax, x1, x2, y, h, text):
    """Dessine une barre d'annotation statistique avec du texte."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

# ==============================================================================
# 4. CRÉATION DU GRAPHIQUE
# ==============================================================================
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Définition de couleurs spécifiques (Bleu classique pour ICA, Orange profond pour CCA)
couleurs = {"ICA (MUedit)": "#4C72B0", "CCA (Decomp-CCA)": "#DD8452"}

# Tracé du Boxplot (les boîtes)
ax = sns.boxplot(x="Condition", y="MUs_Trouvees", hue="Algorithme", data=df, 
                 palette=couleurs, showmeans=True, 
                 meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize": 8})

# Tracé du Stripplot (les petits points) avec contours noirs
sns.stripplot(x="Condition", y="MUs_Trouvees", hue="Algorithme", data=df, 
              palette=couleurs, alpha=0.9, dodge=True, ax=ax, legend=False,
              linewidth=1, edgecolor='black', size=6)

# --- CALCUL ET AFFICHAGE DES P-VALUES SUR LE GRAPHIQUE ---
conditions = ['Normal', 'PLI', 'MA', 'SNR_WGN']

# On trouve la hauteur maximale du graphique pour placer les barres au-dessus
y_max = df['MUs_Trouvees'].max()
bar_height = y_max * 0.05 # Hauteur du petit trait vertical

for i, cond in enumerate(conditions):
    # Extraction des données pour le t-test
    data_ica = df[(df['Condition'] == cond) & (df['Algorithme'] == 'ICA (MUedit)')]['MUs_Trouvees']
    data_cca = df[(df['Condition'] == cond) & (df['Algorithme'] == 'CCA (Decomp-CCA)')]['MUs_Trouvees']
    
    # Calcul du t-test de Student indépendant
    t_stat, p_val = stats.ttest_ind(data_ica, data_cca)
    
    # Conversion de la p-value en étoiles
    if p_val < 0.001:
        sig_text = "***"
    elif p_val < 0.01:
        sig_text = "**"
    elif p_val < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"
    
    # Les positions X des deux boîtes (ICA est légèrement à gauche, CCA légèrement à droite)
    x_ica = i - 0.2
    x_cca = i + 0.2
    
    # Dessin de la barre
    y_bar = y_max + (i % 2) * (y_max * 0.1) # Alterne la hauteur pour ne pas superposer les barres
    draw_p_value_bar(ax, x_ica, x_cca, y_bar, bar_height, sig_text)

# ==============================================================================
# 5. FINITIONS VISUELLES
# ==============================================================================
plt.title('Comparaison des performances : ICA (MUedit) vs CCA (Decomp-CCA)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Nombre d\'Unités Motrices Validées', fontsize=12, fontweight='bold')
plt.xlabel('Condition de Simulation (Bruit)', fontsize=12, fontweight='bold')

# Augmenter la limite Y pour faire de la place aux barres de significativité
plt.ylim(0, y_max + (y_max * 0.3))

plt.legend(title='Algorithme d\'Extraction', loc='lower left')
plt.tight_layout()

# Affichage du graphique
plt.show()

# ==============================================================================
# 6. EXPORT DU RAPPORT DE SYNTHÈSE (Console)
# ==============================================================================
print("\n" + "="*50)
print("=== RAPPORT DE SYNTHÈSE DES PERFORMANCES ===")
print("="*50)

# Calcul des moyennes et écarts-types par condition et par algorithme
resume_stats = df.groupby(['Condition', 'Algorithme'])['MUs_Trouvees'].agg(['mean', 'std']).round(2)

# Renommer les colonnes pour que ce soit propre
resume_stats.columns = ['Moyenne_UM', 'Ecart_Type_UM']

print("\n--- STATISTIQUES DESCRIPTIVES ---")
print(resume_stats.to_string())
print("\n" + "="*50 + "\n")