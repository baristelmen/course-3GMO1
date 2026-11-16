"""Modèles de la partie II : système de Lorenz, système sans dissipation,
oscillateur harmonique (sections 2, 5 et 6 de l'énoncé).
Convention : f(t, y) renvoie dy/dt ; une trajectoire y_res est un tableau
(3, Nt), lignes X, Y, Z, colonne n à l'instant t_n = n*h."""

import numpy as np


def systeme_dynamique(t, y, Pr, r, b):
    # Second membre du système de Lorenz.
    # y peut être de forme (3,) ou (3, M) : r peut alors être un tableau (M,)
    # (broadcasting, utilisé par integration_temporelle_vectorisee)
    X = y[0]
    Y = y[1]
    Z = y[2]
    dy = np.array([
        Pr * (Y - X),
        -X*Z + r*X - Y,
        X*Y - b*Z,
    ])
    return dy


def jacobienne_lorenz(t, y, Pr, r, b):
    # Matrice jacobienne J[i, j] = d f_i / d y_j du système de Lorenz
    # (stabilité des points fixes, schémas implicites)
    X = y[0]
    Y = y[1]
    Z = y[2]
    J = np.array([
        [-Pr, Pr, 0],
        [r - Z, -1, -X],
        [Y, X, -b],
    ])
    return J


def points_fixes(r, b):
    # Solutions de f(y) = 0 : l'origine O, puis C+ et C- si r > 1
    pts = [np.array([0.0, 0.0, 0.0])]
    if r > 1:
        a = np.sqrt(b * (r - 1))
        pts.append(np.array([a, a, r - 1]))      # C+
        pts.append(np.array([-a, -a, r - 1]))    # C-
    return pts


def systeme_conservatif(t, y, Pr, r):
    # Système de Lorenz sans les termes de dissipation -Pr X, -Y et -b Z
    # (section 6.1) : il conserve C1 = Y^2 + Z^2 - 2 r Z et C2 = X^2 - 2 Pr Z
    X = y[0]
    Y = y[1]
    Z = y[2]
    dy = np.array([
        Pr * Y,
        -X*Z + r*X,
        X*Y,
    ])
    return dy


def jacobienne_conservatif(t, y, Pr, r):
    # Matrice jacobienne du système sans dissipation
    X = y[0]
    Y = y[1]
    Z = y[2]
    J = np.array([
        [0, Pr, 0],
        [r - Z, 0, -X],
        [Y, X, 0],
    ])
    return J


def invariants_conservatif(y_res, Pr, r):
    # Invariants C1 et C2 le long d'une trajectoire y_res (3, Nt)
    C1 = y_res[1]**2 + y_res[2]**2 - 2*r*y_res[2]
    C2 = y_res[0]**2 - 2*Pr*y_res[2]
    return C1, C2


def oscillateur_harmonique(t, y):
    # x'' + x = 0 à l'ordre 1 : y = (x, v), dx/dt = v, dv/dt = -x
    # (valeurs propres +-i, énergie E = (x^2 + v^2)/2 conservée)
    return np.array([y[1], -y[0]])


def jacobienne_oscillateur(t, y):
    # Matrice jacobienne de l'oscillateur harmonique (constante)
    return np.array([[0.0, 1.0], [-1.0, 0.0]])
