%% SCRIPT DE CONVERSION FINAL - CONFORME AU MANUEL MUEDIT (Page 6)
clear; clc;

% --- 1. Paramètres du fichier ---
source_file = 'contraction_bb.Poly5'; % Fichier brut
fsamp = 2048;                        % Fréquence d'échantillonnage (Mobita)
nChan = 64;                          % Nombre de canaux de ta grille

% --- 2. Lecture du binaire (Extraction des données) ---
if ~exist(source_file, 'file')
    error('Le fichier %s est introuvable.', source_file);
end

fid = fopen(source_file, 'r');
fseek(fid, 4096, 'bof'); % On saute l'entête textuelle du format Poly5
raw_data = fread(fid, 'int16'); 
fclose(fid);

% --- 3. Organisation de la matrice (signal.data) ---
nb_points = length(raw_data);
nb_col = floor(nb_points / nChan);
data_matrix = reshape(raw_data(1:nChan * nb_col), nChan, nb_col);

% --- 4. Création de la structure 'signal' (Obligatoire pour MUedit) ---
signal.data = double(data_matrix);     % Format 'double' requis par le manuel
signal.fsamp = fsamp;                  % Fréquence d'échantillonnage
signal.nChan = nChan;                  % Nombre de canaux
signal.ngrid = 1;                      % 1 seule grille utilisée

% IMPORTANT : gridname doit être une cellule {} et le nom doit exister dans MUedit
signal.gridname = {'GR08MM1305'};      % Correspond à la grille 13x5 du manuel

% IMPORTANT : muscle doit aussi être une cellule {}
signal.muscle = {'Biceps Brachii'};

% --- 5. Sauvegarde ---
output_name = 'mon_biceps_muedit.mat';
save(output_name, 'signal', '-v7');

fprintf('Félicitations ! Le fichier "%s" est prêt pour MUedit.\n', output_name);