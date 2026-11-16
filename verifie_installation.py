# Chaos temporel en convection de Rayleigh-Bénard -- partie II
# Vérification de l'installation (Windows, macOS, Linux).
# Lancement : python verifie_installation.py   (ou bouton ▷ de VSCode)
# Affiche les versions des bibliothèques, vérifie que les versions de RK4
# donnent le même résultat et mesure leur vitesse.

import sys
import time
import importlib

# %% Bibliothèques nécessaires (et facultatives)
print("Python", sys.version.split()[0])
manquantes = []
for nom, indispensable in [("numpy", True), ("scipy", True), ("matplotlib", True),
                           ("numba", False), ("sklearn", False), ("ipykernel", False)]:
    try:
        module = importlib.import_module(nom)
        print("  %-11s %s" % (nom, getattr(module, "__version__", "?")))
    except ImportError:
        print("  %-11s ABSENT (%s)" % (nom, "indispensable" if indispensable else
                                        "conseillé : numba accélère les calculs, sklearn "
                                        "sert à la section 7, ipykernel au notebook"))
        if indispensable:
            manquantes.append(nom)
if manquantes:
    print("Installez d'abord :", " ".join(manquantes), "(voir l'énoncé, section 1)")
    sys.exit(1)

# %% Les versions de RK4 donnent-elles le même résultat ?
import numpy as np
from integrateurs import (NUMBA_DISPONIBLE, integration_temporelle_python,
                          integration_temporelle, integration_temporelle_numba)

Nt = 5001
h = 0.01
y_numpy = integration_temporelle([1, 1, 1], Nt, 10, 28, 8/3, h)
y_python = integration_temporelle_python([1, 1, 1], Nt, 10, 28, 8/3, h)
debut = time.perf_counter()
y_numba = integration_temporelle_numba([1, 1, 1], Nt, 10, 28, 8/3, h)
duree_compilation = time.perf_counter() - debut
print("Écart pur Python / boucle NumPy : %.1e" % np.max(np.abs(y_python - y_numpy)))
print("Écart Numba / boucle NumPy      : %.1e" % np.max(np.abs(y_numba - y_numpy)))
if NUMBA_DISPONIBLE:
    print("Numba : premier appel (compilation) %.1f s" % duree_compilation)
else:
    print("Numba absent : les calculs marchent, mais plus lentement")

# %% Vitesse
for nom, fonction in [("pur Python", integration_temporelle_python),
                      ("boucle NumPy", integration_temporelle),
                      ("Numba", integration_temporelle_numba)]:
    debut = time.perf_counter()
    fonction([1, 1, 1], Nt, 10, 28, 8/3, h)
    print("  %-12s %6.2f µs par pas" % (nom, (time.perf_counter() - debut) / (Nt - 1) * 1e6))
print("Installation correcte.")
