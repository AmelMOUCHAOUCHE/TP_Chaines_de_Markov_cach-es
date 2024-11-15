import numpy as np
import pandas as pd
import re

def lire_corpus(nom_fichier):
    """
    Lit un fichier texte contenant un corpus de mots et retourne une liste de mots nettoyés.
    
    Args:
        nom_fichier (str): Le chemin du fichier texte.
    
    Returns:
        list: Une liste de mots nettoyés, en minuscules et sans caractères spéciaux.
    """
    with open(nom_fichier, 'r', encoding='utf-8') as fichier:
        lignes = fichier.readlines()
    
    mots_nettoyes = []
    
    for ligne in lignes:
        ligne = ligne.lower()
        
        mot_nettoye = ''.join([caractere for caractere in ligne if 'a' <= caractere <= 'z'])
        
        if mot_nettoye:
            mots_nettoyes.append(mot_nettoye)
            
    return mots_nettoyes

def matrice_emission(nom_fichier):
    """
    Lit une matrice d'émission depuis un fichier Excel et la convertit en une matrice NumPy.
    
    Args:
        nom_fichier (str): Le chemin du fichier Excel contenant la matrice d'émission.
    
    Returns:
        np.ndarray: La matrice d'émission B.
    """
    df = pd.read_excel(nom_fichier, index_col=0)
    B = df.to_numpy()
    return B

def matrice_transition(nom_fichier):
    """
    Génère la matrice de transition à partir d'un corpus de mots.
    
    Args:
        nom_fichier (str): Le chemin du fichier texte contenant le corpus de mots.
    
    Returns:
        np.ndarray: La matrice de transition A.
    """
    A = np.zeros((26, 26))  # 26 lettres a-z
    
    corpus = lire_corpus(nom_fichier)
    
    for mot in corpus:
        indices = [ord(c) - ord('a') for c in mot if 'a' <= c <= 'z']
        for i in range(len(indices) - 1):
            A[indices[i], indices[i + 1]] += 1
    
    for i in range(26):
        somme_ligne = np.sum(A[i])
        if somme_ligne > 0:
            A[i] /= somme_ligne  # Normalisation des lignes
    
    return A

def creer_modele_hmm(nom_fichier_corpus, nom_fichier_emission):
    """
    Crée un modèle HMM pour une langue donnée à partir d'un corpus de mots et d'une matrice d'émission.
    
    Args:
        nom_fichier_corpus (str): Chemin du fichier texte contenant le corpus de mots.
        nom_fichier_emission (str): Chemin du fichier Excel contenant la matrice d'émission.
    
    Returns:
        tuple: Le modèle HMM (pi, A, B) pour la langue donnée.
    """
    A = matrice_transition(nom_fichier_corpus)
    B = matrice_emission(nom_fichier_emission)
    pi = np.ones(26) / 26  # Probabilité initiale uniforme
    
    return pi, A, B


def forward(O, A, B, PI):
    """
    Effectue l'algorithme forward pour calculer la probabilité d'observer une séquence d'observations O
    étant donné le modèle HMM (A, B, PI).

    Args:
        O (list): La séquence d'observations (représentée par des indices de lettres).
        A (np.ndarray): La matrice de transition (taille NxN, où N est le nombre d'états).
        B (np.ndarray): La matrice d'émission (taille NxM, où M est le nombre d'observations possibles).
        PI (np.ndarray): Le vecteur des probabilités initiales (taille N).

    Returns:
        P (float): La probabilité d'observer la séquence O donnée le modèle HMM.
        alpha (np.ndarray): La matrice alpha, où alpha[t, i] est la probabilité d'être dans l'état i après avoir observé les premières t observations.
    """
    N = len(PI)  # Nombre d'états
    T = len(O)   # Longueur de la séquence d'observations

    # Initialisation de la matrice alpha
    alpha = np.zeros((T, N))

    # Calcul de alpha[0, i] pour chaque état i
    for i in range(N):
        alpha[0, i] = PI[i] * B[i, O[0]]

    # Calcul de alpha[t, j] pour t > 0
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, O[t]]

    # La probabilité totale d'observer la séquence O est la somme de alpha[T-1, :]
    P = np.sum(alpha[T-1, :])

    return P, alpha


def backward(O, A, B, PI):
    """
    Implémente l'algorithme backward pour calculer P(O|λ) et la variable beta.
    
    :param O: La séquence d'observations (série de symboles observés)
    :param A: La matrice de transition (A[i, j] est la probabilité de transition de l'état i vers l'état j)
    :param B: La matrice d'émission (B[j, k] est la probabilité d'émettre l'observation k à partir de l'état j)
    :param PI: La distribution initiale des états (PI[i] est la probabilité que l'état initial soit i)
    :return: La probabilité P(O|λ) et la variable beta (tableau des probabilités cumulées à l'envers)
    """
    N = A.shape[0]  # Nombre d'états
    T = len(O)      # Longueur de la séquence d'observations
    
    # Initialisation de beta (tableau des probabilités cumulées à l'envers)
    beta = np.zeros((T, N))
    
    # Étape 1 : Initialisation (au dernier instant T-1)
    beta[T-1, :] = 1  # Les probabilités de la dernière observation sont égales à 1
    
    # Étape 2 : Récursion (Induction) en partant de la fin de la séquence
    for t in range(T-2, -1, -1):  # On parcourt les observations en sens inverse
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, O[t+1]] * beta[t+1, :])
    
    # Étape 3 : Terminaison
    # La probabilité totale P(O|λ) est la somme de alpha[0, :] * beta[0, :]
    P = np.sum(PI * B[:, O[0]] * beta[0, :])
    return P, beta


def lire_fichier(nom_fichier):
    text = ""
    with open(nom_fichier, 'r') as f: 
        for l in f.readlines():
            text += l
    O = text.split()
    O = [re.sub('[^a-z]', '', m.lower()) for m in O]
    # Suppression des chaînes vides
    O = [m for m in O if m] 
    #O = text.lower()
    #O = re.sub('[^a-z]', '', O)
    #O = text.split()
    return O
