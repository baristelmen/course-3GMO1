"""Jeux de données de la section 7 (apprentissage automatique) de la partie II.
Les fichiers data/amplitudes.json, data/bassins_r10.json et data/bassins_r20.json
sont fournis : il n'est pas nécessaire de les recalculer (environ 1 min avec
numba, beaucoup plus sans). Les fonctions ci-dessous montrent comment ils ont
été produits."""

import json

import numpy as np

from analyse import DOSSIER_DONNEES
from integrateurs import integration_temporelle_numba


def calcul_amplitudes(Pr, b, N_r=200, tmax=300, h=0.01):
    # Amplitude A(r) = |X(tmax)| de la convection stationnaire, depuis [1, 1, 1],
    # pour N_r valeurs de r dans [1.05, 23]
    rs = np.linspace(1.05, 23, N_r)
    Nt = int(round(tmax / h)) + 1
    A = [abs(integration_temporelle_numba([1, 1, 1], Nt, Pr, r, b, h)[0, -1])
         for r in rs]
    return {"description": "Amplitude de la convection stationnaire |X|",
            "methode": "RK4, h = %g, tmax = %g, y0 = [1, 1, 1]" % (h, tmax),
            "r": rs.tolist(), "A": [float(a) for a in A]}


def calcul_bassins(Pr, r, b, N_points=4000, tmax=150, h=0.01, graine=0):
    # Bassins d'attraction de C+ (+1) et C- (-1) : N_points conditions
    # initiales (X0, Y0) tirées dans [-30, 30]^2, avec Z0 = r - 1
    rng = np.random.default_rng(graine)
    XY = rng.uniform(-30, 30, (N_points, 2))
    Nt = int(round(tmax / h)) + 1
    etiquettes = []
    for X0, Y0 in XY:
        y_fin = integration_temporelle_numba([X0, Y0, r - 1], Nt, Pr, r, b, h)[:, -1]
        etiquettes.append(int(np.sign(y_fin[0])))
    return {"description": "Bassins d'attraction de C+ (+1) et C- (-1)",
            "methode": "RK4, h = %g, tmax = %g" % (h, tmax), "r": r,
            "Z0": r - 1, "X0": XY[:, 0].tolist(), "Y0": XY[:, 1].tolist(),
            "etiquette": etiquettes}


def charge_donnees(nom):
    # Lecture de data/<nom>.json (nom = "amplitudes", "bassins_r10", "bassins_r20")
    with open(DOSSIER_DONNEES / (nom + ".json"), encoding="utf-8") as fichier:
        return json.load(fichier)
