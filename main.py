import numpy as np
from mylib import lire_corpus, matrice_emission, matrice_transition, creer_modele_hmm, forward, backward, lire_fichier

    
alphabet = [chr(i) for i in range(97, 123)]  # Alphabet a-z

# Chargement des corpus
corpus_fr = lire_corpus('french.txt')
corpus_en = lire_corpus('english.txt')
corpus_it = lire_corpus('italian.txt')

print(f"Nombre de mots dans le corpus français : {len(corpus_fr)}")
print(f"Nombre de mots dans le corpus anglais : {len(corpus_en)}")
print(f"Nombre de mots dans le corpus italien : {len(corpus_it)}")

# Chargement de la matrice d'émission à partir du fichier Excel
nom_fichier_emission = 'matrice_emission.xls'
matrice_B = matrice_emission(nom_fichier_emission)

# Affichage de la matrice d'émission pour vérification
print("Matrice d'émission B :")
print(matrice_B)

# Exemple : Charger la matrice de transition pour le corpus français
nom_fichier_corpus = 'french.txt'
matrice_A = matrice_transition(nom_fichier_corpus)

# Afficher la matrice de transition pour vérification
print("Matrice de transition A :")
print(matrice_A)

# Créer les modèles HMM pour chaque langue
pi_fr, A_fr, B_fr = creer_modele_hmm(nom_fichier_corpus, nom_fichier_emission)
pi_en, A_en, B_en = creer_modele_hmm('english.txt', nom_fichier_emission)
pi_it, A_it, B_it = creer_modele_hmm('italian.txt', nom_fichier_emission)

# Affichage des résultats
print("\nModèle HMM pour le français :")
print("pi :", pi_fr)
print("A :\n", A_fr)
print("B :\n", B_fr)

print("\nModèle HMM pour l'anglais :")
print("pi :", pi_en)
print("A :\n", A_en)
print("B :\n", B_en)

print("\nModèle HMM pour l'italien :")
print("pi :", pi_it)
print("A :\n", A_it)
print("B :\n", B_it)

# Exemple d'utilisation de la fonction forward
O = [0, 1]  # Exemple de séquence d'observations (indices des lettres)

# Matrices d'exemple (A, B, PI)
A = np.array([[0.7, 0.3], [0.4, 0.6]])  # Matrice de transition
B = np.array([[0.5, 0.5], [0.1, 0.9]])  # Matrice d'émission
PI = np.array([0.6, 0.4])  # Probabilité initiale

# Calcul de la probabilité d'observer la séquence O
P, alpha = forward(O, A, B, PI)
print("Probabilité d'observer O :", P)
print("Matrice alpha :", alpha)

# Calcul avec une autre séquence d'observations
O = [0, 1, 0]
P_forward, alpha = forward(O, A, B, PI)
print("Probabilité (forward) :", P_forward)

P_backward, beta = backward(O, A, B, PI)
print("Probabilité (backward) :", P_backward)

# Liste des fichiers texte à analyser
fichiers_textes = ['texte_1.txt', 'texte_2.txt', 'texte_3.txt']

# Fonction pour convertir un mot en une séquence d'indices
def convertir_en_indices(mot):
    return [ord(c) - ord('a') for c in mot if 'a' <= c <= 'z']

# Dictionnaire pour stocker les probabilités
probabilites = {'FR': [], 'EN': [], 'IT': []}

# Calcul de la probabilité pour chaque mot
for fichier in fichiers_textes:
    mots = lire_fichier(fichier)
    O = [ord(c) - ord('a') for mot in mots for c in mot]  # Conversion en indices
    
    # Calcul des probabilités pour chaque langue
    P_fr, _ = forward(O, A_fr, B_fr, pi_fr)
    P_en, _ = forward(O, A_en, B_en, pi_en)
    P_it, _ = forward(O, A_it, B_it, pi_it)
    
    # Affichage des résultats
    print(f"Probabilités pour {fichier} :")
    print(f"  P({fichier} | λ_FR) = {P_fr}")
    print(f"  P({fichier} | λ_EN) = {P_en}")
    print(f"  P({fichier} | λ_IT) = {P_it}")
    
    # Déterminer la langue la plus probable
    langue_probable = max((P_fr, 'FR'), (P_en, 'EN'), (P_it, 'IT'))[1]
    print(f"Langue la plus probable : {langue_probable}\n")

# Liste des mots à analyser pour une autre comparaison
mots = ['probablement', 'probably', 'probabilmente']

# Initialisation des modèles HMM pour chaque langue
pi_fr, A_fr, B_fr = creer_modele_hmm('french.txt', 'matrice_emission.xls')
pi_en, A_en, B_en = creer_modele_hmm('english.txt', 'matrice_emission.xls')
pi_it, A_it, B_it = creer_modele_hmm('italian.txt', 'matrice_emission.xls')

# Calcul des probabilités pour chaque mot
for mot in mots:
    O = convertir_en_indices(mot)
    
    # Calcul des probabilités avec l'algorithme forward
    P_fr, _ = forward(O, A_fr, B_fr, pi_fr)
    P_en, _ = forward(O, A_en, B_en, pi_en)
    P_it, _ = forward(O, A_it, B_it, pi_it)
    
    # Affichage des probabilités
    print(f"Probabilités pour le mot '{mot}' :")
    print(f"  P({mot} | λ_FR) = {P_fr}", P_fr)
    print(f"  P({mot} | λ_EN) = {P_en}")
    print(f"  P({mot} | λ_IT) = {P_it}")
    
    # Utilisation de l'algorithme backward pour confirmer les résultats
    P_fr_backward, _ = backward(O, A_fr, B_fr, pi_fr)
    P_en_backward, _ = backward(O, A_en, B_en, pi_en)
    P_it_backward, _ = backward(O, A_it, B_it, pi_it)
    
    print(f"Probabilités (backward) pour le mot '{mot}' :")
    print(f"  P({mot} | λ_FR) = {P_fr_backward}")
    print(f"  P({mot} | λ_EN) = {P_en_backward}")
    print(f"  P({mot} | λ_IT) = {P_it_backward}")

# Comparaison des probabilités et détermination de la langue la plus probable pour chaque mot
for i, mot in enumerate(mots):
    max_prob = max((probabilites['FR'][i], 'FR'), (probabilites['EN'][i], 'EN'), (probabilites['IT'][i], 'IT'))
    print(f"La langue la plus probable pour '{mot}' : {max_prob[1]}")


