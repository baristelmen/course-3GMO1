# Chaos temporel en convection de Rayleigh-Bénard -- partie II
# Calcul des solutions de référence et des données du dossier data.
# Inutile de le relancer : les fichiers sont fournis (environ 1 min avec numba).

import json

import numpy as np

from analyse import solution_reference, sauve_reference_json, DOSSIER_DONNEES
from donnees_apprentissage import calcul_amplitudes, calcul_bassins

# %% Définition des paramètres
Pr = 10
b = 8/3
sec_references = True
sec_donnees_apprentissage = True

# %% Solutions de référence (DOP853)
if sec_references:
    # r = 10 (non chaotique) ; r10_precise : tolérances encore plus sévères
    # (section 4.2) ; r = 28 (chaos), sur [0, 30]
    cas = {"r10": dict(r=10, tmax=2, dt_sortie=2.0**-12),
           "r10_precise": dict(r=10, tmax=2, dt_sortie=2.0**-12, rtol=2.3e-14, atol=1e-15),
           "r28": dict(r=28, tmax=30, dt_sortie=2.0**-10)}
    for nom, p in cas.items():
        ref = solution_reference([1, 1, 1], p["tmax"], p["dt_sortie"], Pr, p["r"], b,
                                 rtol=p.get("rtol", 1e-13), atol=p.get("atol", 1e-13))
        sauve_reference_json(ref, DOSSIER_DONNEES / ("reference_%s.json" % nom))
        print("Référence %s : %d appels à f, %.2f s" % (nom, ref["nfev"], ref["duree_calcul_s"]))

# %% Données de la section 7 (apprentissage automatique)
if sec_donnees_apprentissage:
    donnees = {"amplitudes": calcul_amplitudes(Pr, b),
               "bassins_r10": calcul_bassins(Pr, 10, b),
               "bassins_r20": calcul_bassins(Pr, 20, b)}
    for nom, contenu in donnees.items():
        with open(DOSSIER_DONNEES / (nom + ".json"), "w", encoding="utf-8") as f:
            json.dump(contenu, f)
        print("Données enregistrées :", nom)
