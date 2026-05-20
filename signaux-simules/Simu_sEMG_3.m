%% PIPELINE DE GÉNÉRATION DE 20 SIGNAUX POUR MUEDIT
% Génère exactement 20 signaux répartis en 4 catégories de 5 répétitions,
% au format structure 'signal' requis pour l'ICA de MUedit.

clear; clc;

% =========================================================================
% 1. PARAMÈTRES DE CONFIGURATION (À AJUSTER SELON SYNCHRO)
% =========================================================================
fs = 2048;                  % Fréquence d'échantillonnage (Hz)
muscle_name = {'Rectus Femoris'}; 
grid_type = {'GR08MM1305'}; % Grille HD-sEMG par défaut (64 canaux)
n_channels = 64;            % Minimum 32 requis pour MUedit surface
duree_seconds = 10;         
n_samples = duree_seconds * fs;

% Dossier de sauvegarde
output_dir = './Base_20_MUedit/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% =========================================================================
% 2. CHARGEMENT DU SIGNAL PROPRE DE BASE (SORTIE DE SYNCHRO)
% =========================================================================
% REMPLACE cette matrice par ton signal brut issu de ta simulation SYNCHRO
fprintf('Génération du signal de base via SYNCHRO...\n');
data_propre_synchro = randn(n_channels, n_samples) * 0.1; 

% Vecteur temporel
t = (0:n_samples-1) / fs;

% Niveaux de SNR prévus pour la catégorie Bruit
snr_levels = [3, 5, 11, 15, 20]; 

% =========================================================================
% 3. BOUCLE DE GÉNÉRATION DES 20 SIGNAUX
% =========================================================================
categories = {'Normal', 'PLI', 'MA', 'SNR_WGN'};
total_fichiers = 0;

for c = 1:length(categories)
    condition = categories{c};
    
    for rep = 1:5
        % Reset du signal à chaque itération
        data_modifie = data_propre_synchro; 
        nom_fichier_specifique = sprintf('%s_rep%d', condition, rep);
        
        switch condition
            case 'Normal'
                % Signal pur, aucune modification
                
            case 'PLI'
                % Interférence ligne électrique 50Hz + harmoniques
                for ch = 1:n_channels
                    pli_signal = 0.02 * sin(2*pi*50*t) + 0.01 * sin(2*pi*100*t) + 0.005 * sin(2*pi*150*t);
                    data_modifie(ch, :) = data_modifie(ch, :) + pli_signal;
                end
                
            case 'MA'
                % Artefact de mouvement involontaire (Dérive de ligne de base + Secousse)
                [b, a] = butter(4, 2/(fs/2), 'low'); % Filtre passe-bas à 2Hz pour simuler la dérive
                for ch = 1:n_channels
                    % A. Dérive lente
                    derive = filtfilt(b, a, randn(1, n_samples));
                    derive = (derive / max(abs(derive))) * (0.15 * max(abs(data_modifie(ch, :))));
                    
                    % B. Secousse musculaire involontaire transitoire (0.5 secondes)
                    secousse = zeros(1, n_samples);
                    idx_start = round(n_samples / 3); % Déclenchement au premier tiers
                    idx_end = idx_start + round(0.5 * fs) - 1;
                    forme_secousse = sin(linspace(0, pi, idx_end - idx_start + 1));
                    secousse(idx_start:idx_end) = forme_secousse * (0.4 * max(abs(data_modifie(ch, :))));
                    
                    data_modifie(ch, :) = data_modifie(ch, :) + derive + secousse;
                end
                
            case 'SNR_WGN'
                % On applique un niveau de décibel différent pour chacune des 5 répétitions
                current_snr = snr_levels(rep); 
                nom_fichier_specifique = sprintf('SNR_%ddB', current_snr);
                
                for ch = 1:n_channels
                    data_modifie(ch, :) = awgn(data_modifie(ch, :), current_snr, 'measured');
                end
        end
        
        % =================================================================
        % 4. FORMATAGE STRUCTURÉ REQUIS PAR MUEDIT
        % =================================================================
        signal = struct();
        signal.data      = data_modifie;       % Matrice [Canaux x Temps]
        signal.fsamp     = fs;                 % Fréquence d'échantillonnage
        signal.nChan     = n_channels;         % Nombre de canaux
        signal.ngrid     = 1;                  % 1 seule grille (Rectus Femoris)
        signal.gridname  = grid_type;          % Nom de configuration de la grille
        signal.muscle    = muscle_name;        % Nom du muscle ciblé
        
        % Variables optionnelles d'affichage
        signal.target    = zeros(1, n_samples); 
        signal.path      = zeros(1, n_samples); 

        % =================================================================
        % 5. SAUVEGARDE
        % =================================================================
        filename = sprintf('%s%s.mat', output_dir, nom_fichier_specifique);
        save(filename, 'signal');
        total_fichiers = total_fichiers + 1;
    end
end

fprintf('\nSuccès ! %d fichiers .mat ont été correctement formatés dans "%s"\n', total_fichiers, output_dir);