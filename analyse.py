"""Outils d'analyse de la partie II (sections 2 à 6 de l'énoncé) : solutions de
référence, maxima, corrélation, stabilité des schémas, seuils, enveloppes.
Seules les fonctions sans équivalent direct dans NumPy/SciPy sont écrites ici :
les normes s'obtiennent avec np.linalg.norm, les pentes avec np.polyfit, les
maxima locaux avec scipy.signal.find_peaks, les moyennes avec np.mean."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.signal import find_peaks

from lorenz import systeme_dynamique, jacobienne_lorenz, points_fixes
from integrateurs import integration_temporelle_numba

# Dossier des solutions de référence et des données (.json)
DOSSIER_DONNEES = Path(__file__).resolve().parent / "data"


# %% Solutions de référence (DOP853) et fichiers .json

def solution_reference(y0, tmax, dt_sortie, Pr, r, b, rtol=1e-13, atol=1e-13,
                       methode="DOP853"):
    # Solution de référence sur [0, tmax], enregistrée tous les dt_sortie, par
    # scipy.integrate.solve_ivp (DOP853 : Runge-Kutta explicite d'ordre 8, pas
    # adaptatif). Choisir dt_sortie = 2^-k : les instants n*h des simulations
    # avec h = 2^-j tombent alors exactement sur la grille de la référence
    Nt = int(round(tmax / dt_sortie)) + 1
    ts = np.linspace(0, tmax, Nt)
    options = {}
    if methode in ("Radau", "BDF", "LSODA"):
        options["jac"] = lambda t, y: jacobienne_lorenz(t, y, Pr, r, b)
    debut = time.perf_counter()
    sol = solve_ivp(lambda t, y: systeme_dynamique(t, y, Pr, r, b), (0, tmax),
                    np.asarray(y0, dtype=float), method=methode, t_eval=ts,
                    rtol=rtol, atol=atol, **options)
    return {"methode": methode, "rtol": rtol, "atol": atol,
            "parametres": {"Pr": Pr, "r": r, "b": b},
            "y0": [float(v) for v in y0], "tmax": tmax, "dt_sortie": dt_sortie,
            "nfev": int(sol.nfev), "duree_calcul_s": time.perf_counter() - debut,
            "ts": sol.t, "y_ref": sol.y}


def sauve_reference_json(ref, chemin):
    # Écriture dans un fichier texte .json (module json de Python)
    contenu = {cle: valeur for cle, valeur in ref.items() if cle not in ("ts", "y_ref")}
    contenu["ts"] = ref["ts"].tolist()
    contenu["X"], contenu["Y"], contenu["Z"] = (ref["y_ref"][i].tolist() for i in range(3))
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(contenu, fichier)


def charge_reference_json(nom):
    # Lecture d'une référence du dossier data : "r10", "r10_precise", "r28"
    with open(DOSSIER_DONNEES / ("reference_%s.json" % nom), encoding="utf-8") as fichier:
        ref = json.load(fichier)
    ref["ts"] = np.array(ref["ts"])
    ref["y_ref"] = np.array([ref.pop("X"), ref.pop("Y"), ref.pop("Z")])
    return ref


def reference_aux_instants(ref, ts):
    # Valeurs de la référence aux instants ts (tableau (3, len(ts))) :
    # interpolation d'Hermite cubique (scipy.interpolate.CubicHermiteSpline),
    # qui utilise y ET sa dérivée f(y) ; erreur d'interpolation ~1e-12
    p = ref["parametres"]
    dy = systeme_dynamique(0, ref["y_ref"], p["Pr"], p["r"], p["b"])
    spline = CubicHermiteSpline(ref["ts"], ref["y_ref"], dy, axis=1)
    if np.max(ts) > ref["ts"][-1] + 1e-12:
        raise ValueError("la référence ne couvre que [0, %g]" % ref["ts"][-1])
    return spline(ts)


# %% Maxima de Z(t)

def maxima_Z_parabole(y_res, ts):
    # Maxima locaux des échantillons Z_n (scipy.signal.find_peaks), puis sommet
    # de la parabole passant par Z_{n-1}, Z_n, Z_{n+1} : pour a, c, d ces trois
    # valeurs, le sommet est en t_n + delta*h, delta = (a - d) / (2 (a - 2c + d)),
    # et vaut c - (a - d) delta / 4 (le sommet remplace l'échantillon Z_n)
    Z = y_res[2]
    h = ts[1] - ts[0]
    ii, _ = find_peaks(Z)
    a, c, d = Z[ii-1], Z[ii], Z[ii+1]
    delta = (a - d) / (2*(a - 2*c + d))
    return c - (a - d)*delta/4, ts[ii] + delta*h


def maxima_Z_hermite(y_res, ts, b):
    # On utilise aussi dZ/dt = X Y - b Z (calculable à partir de l'état) : sur
    # chaque pas, CubicHermiteSpline construit le polynôme de degré 3 qui a les
    # bonnes valeurs ET les bonnes dérivées ; les maxima sont les zéros de sa
    # dérivée où la dérivée seconde est négative
    dZ = y_res[0]*y_res[1] - b*y_res[2]
    spline = CubicHermiteSpline(ts, y_res[2], dZ)
    derivee = spline.derivative()
    ts_max = derivee.roots(extrapolate=False)
    ts_max = ts_max[(ts_max > ts[0]) & (ts_max < ts[-1])]
    ts_max = ts_max[derivee.derivative()(ts_max) < 0]
    return spline(ts_max), ts_max


# %% Régime chaotique

def correlation_glissante(ts, signal_a, signal_b, largeur):
    # Coefficient de corrélation (np.corrcoef) entre deux signaux, sur des
    # fenêtres successives de durée largeur : 1 si les signaux sont identiques
    # (à un facteur près), environ 0 s'ils n'ont plus de lien
    n = max(int(round(largeur / (ts[1] - ts[0]))), 2)
    centres, rho = [], []
    for debut in range(0, len(ts) - n + 1, n):
        centres.append(ts[debut + n//2])
        rho.append(np.corrcoef(signal_a[debut:debut+n], signal_b[debut:debut+n])[0, 1])
    return np.array(centres), np.array(rho)


def points_sur_attracteur(M, Pr, r, b, graine=0):
    # M conditions initiales SUR l'attracteur : longue trajectoire (RK4,
    # h = 0.01), transitoire oublié, puis M points tirés au hasard
    rng = np.random.default_rng(graine)
    y_res = integration_temporelle_numba([1, 1, 1], 20001, Pr, r, b, 0.01)[:, 5000:]
    return y_res[:, rng.choice(y_res.shape[1], M, replace=False)].T


# %% Stabilité des schémas et des points fixes

def fonction_stabilite(nom_schema, z):
    # Fonction de stabilité R(z) : un pas appliqué à dy/dt = lambda y donne
    # y_{n+1} = R(h lambda) y_n (z = h lambda complexe). Pour BDF2 (deux pas) :
    # racine de plus grand module de (1 - 2z/3) s^2 - 4/3 s + 1/3 = 0
    z = np.asarray(z, dtype=complex)
    if nom_schema == "Euler explicite":
        return 1 + z
    if nom_schema == "Heun (RK2)":
        return 1 + z + z**2/2
    if nom_schema == "RK4":
        return 1 + z + z**2/2 + z**3/6 + z**4/24
    if nom_schema == "Euler implicite":
        return 1 / (1 - z)
    if nom_schema in ("Crank-Nicolson", "Point milieu"):
        return (1 + z/2) / (1 - z/2)
    if nom_schema in ("SDIRK2", "TR-BDF2"):
        g = 1 - np.sqrt(2)/2
        return (1 + (1 - 2*g)*z) / (1 - g*z)**2
    if nom_schema == "BDF2":
        zs = np.atleast_1d(z)
        R = np.empty(zs.shape, dtype=complex)
        for indice, zi in np.ndenumerate(zs):
            racines = np.roots([1 - 2*zi/3, -4/3, 1/3])
            R[indice] = racines[np.argmax(np.abs(racines))]
        return R.reshape(z.shape)
    raise ValueError("schéma inconnu : %s" % nom_schema)


def valeurs_propres_C_plus(Pr, r, b):
    # Valeurs propres de la jacobienne au point fixe C+ (r > 1)
    return np.linalg.eigvals(jacobienne_lorenz(0, points_fixes(r, b)[1], Pr, r, b))


def seuil_hopf_numerique(nom_schema, h, Pr, b, r_min=1.5, r_max=40):
    # Valeur de r où C+ perd sa stabilité POUR LE SCHÉMA : max_i |R(h lambda_i(r))|
    # passe par 1 (zéro trouvé par scipy.optimize.brentq). NaN s'il n'y a pas de
    # changement de stabilité sur [r_min, r_max]
    def g(r):
        lambdas = valeurs_propres_C_plus(Pr, r, b)
        return np.max(np.abs(fonction_stabilite(nom_schema, h*lambdas))) - 1
    if np.sign(g(r_min)) == np.sign(g(r_max)):
        return np.nan
    return brentq(g, r_min, r_max, xtol=1e-12)


def taux_enveloppe(signal, ts, tmin=None, tmax=None):
    # Taux de croissance sigma de l'enveloppe d'un signal oscillant : maxima
    # successifs de |signal| (sommets de paraboles), puis ajustement
    # ln(max_k) = sigma t_k + c par np.polyfit.
    # sigma > 0 : oscillations amplifiées ; < 0 : amorties ; = 0 : conservées
    y_res = np.zeros((3, len(signal)))
    y_res[2] = np.abs(signal)
    maxima, ts_max = maxima_Z_parabole(y_res, ts)
    garde = maxima > 0
    if tmin is not None:
        garde &= ts_max >= tmin
    if tmax is not None:
        garde &= ts_max <= tmax
    maxima, ts_max = maxima[garde], ts_max[garde]
    if len(maxima) < 2:
        return np.nan, maxima, ts_max, None
    coefficients = np.polyfit(ts_max, np.log(maxima), 1)
    return coefficients[0], maxima, ts_max, coefficients


def balayage_quasi_statique(rs, y0, Pr, b, tmax=200, h=0.01, bruit=1e-3, graine=0):
    # r varie par paliers (rs, croissant ou décroissant). À chaque palier : on
    # part de l'état final du palier précédent, on ajoute une perturbation
    # aléatoire d'amplitude bruit (bruit expérimental), on intègre pendant tmax
    # (RK4, numba) et on mesure A(r) = max |Z(t) - (r - 1)| sur la seconde moitié
    # du palier (petit sur les rouleaux C+ ou C-, ~20 en régime chaotique)
    rng = np.random.default_rng(graine)
    Nt = int(round(tmax / h)) + 1
    y = np.asarray(y0, dtype=float)
    amplitudes = np.zeros(len(rs))
    etats_finaux = np.zeros((len(rs), 3))
    for ii, r in enumerate(rs):
        y = y + bruit * rng.standard_normal(3)
        y_res = integration_temporelle_numba(y, Nt, Pr, r, b, h)
        amplitudes[ii] = np.max(np.abs(y_res[2, Nt//2:] - (r - 1)))
        y = y_res[:, -1]
        etats_finaux[ii] = y
    return amplitudes, etats_finaux
