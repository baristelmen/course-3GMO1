"""Intégration temporelle (sections 2, 3 et 4 de l'énoncé).
1) Un pas de chaque schéma à pas fixe (annexe A) :
     explicites : ynp1 = integration_XXX(f, yn, t, h)
     implicites : ynp1, nfev = integration_XXX(f, yn, t, h, jac)
   (jac(t, y) : matrice jacobienne de f ; le système non linéaire de chaque pas
   est résolu par scipy.optimize.fsolve ; nfev : nombre d'appels à la fonction)
2) integration_temporelle_schema : intégration complète avec l'un de ces schémas.
3) RK4 pour Lorenz en plusieurs versions (section 3), qui calculent exactement
   la même chose : pur Python, NumPy à boucles, NumPy vectorisé (plusieurs r à
   la fois), Numba ; plus une version en simple précision.
Remarque : scipy.integrate.solve_ivp ne propose que des méthodes à pas
ADAPTATIF (RK23, RK45, DOP853, Radau, BDF, LSODA) ; on ne peut pas y imposer
un pas fixe, ce qui est nécessaire pour mesurer un ordre de convergence."""

import warnings

import numpy as np
from scipy.optimize import fsolve

from lorenz import systeme_dynamique

try:
    from numba import njit
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False

# Paramètres de SDIRK2 et de TR-BDF2 (voir l'annexe A)
GAMMA_SDIRK2 = 1 - np.sqrt(2)/2
GAMMA_TRBDF2 = 2 - np.sqrt(2)


# %% 1) Un pas de chaque schéma

def integration_Euler(f, yn, t, h):
    # Euler explicite : y_{n+1} = y_n + h f(t_n, y_n)
    return yn + h * f(t, yn)


def integration_RK2(f, yn, t, h):
    # Méthode de Heun (Runge-Kutta d'ordre 2)
    k1 = f(t, yn)
    k2 = f(t + h, yn + h*k1)
    return yn + h/2 * (k1 + k2)


def integration_RK4(f, yn, t, h):
    # Runge-Kutta d'ordre 4
    k1 = f(t, yn)
    k2 = f(t + h/2, yn + h/2*k1)
    k3 = f(t + h/2, yn + h/2*k2)
    k4 = f(t + h, yn + h*k3)
    return yn + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def _resout(G, JG, z0):
    # Résolution de G(z) = 0 par scipy.optimize.fsolve (jacobienne JG fournie)
    z, infos, _, _ = fsolve(G, z0, fprime=JG, xtol=1e-12, full_output=True)
    return z, infos["nfev"]


def integration_Euler_implicite(f, yn, t, h, jac):
    # y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})
    Id = np.eye(len(yn))
    return _resout(lambda z: z - yn - h*f(t + h, z),
                   lambda z: Id - h*jac(t + h, z), yn)


def integration_Crank_Nicolson(f, yn, t, h, jac):
    # y_{n+1} = y_n + h/2 (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))   (trapèzes)
    Id = np.eye(len(yn))
    fn = f(t, yn)
    return _resout(lambda z: z - yn - h/2*(fn + f(t + h, z)),
                   lambda z: Id - h/2*jac(t + h, z), yn)


def integration_point_milieu(f, yn, t, h, jac):
    # y_{n+1} = y_n + h f(t_n + h/2, (y_n + y_{n+1})/2)
    Id = np.eye(len(yn))
    return _resout(lambda z: z - yn - h*f(t + h/2, (yn + z)/2),
                   lambda z: Id - h/2*jac(t + h/2, (yn + z)/2), yn)


def integration_SDIRK2(f, yn, t, h, jac):
    # Deux étages implicites de même coefficient gamma (annexe A)
    g = GAMMA_SDIRK2
    Id = np.eye(len(yn))
    k1, n1 = _resout(lambda k: k - f(t + g*h, yn + h*g*k),
                     lambda k: Id - h*g*jac(t + g*h, yn + h*g*k), f(t, yn))
    base = yn + h*(1 - g)*k1
    k2, n2 = _resout(lambda k: k - f(t + h, base + h*g*k),
                     lambda k: Id - h*g*jac(t + h, base + h*g*k), k1)
    return yn + h*((1 - g)*k1 + g*k2), n1 + n2


def integration_TR_BDF2(f, yn, t, h, jac):
    # Un pas de trapèzes jusqu'à t + g h, puis un pas de BDF2 (annexe A)
    g = GAMMA_TRBDF2
    Id = np.eye(len(yn))
    fn = f(t, yn)
    yg, n1 = _resout(lambda z: z - yn - g*h/2*(fn + f(t + g*h, z)),
                     lambda z: Id - g*h/2*jac(t + g*h, z), yn)
    c = (1 - g) / (2 - g)
    second_membre = (yg - (1 - g)**2 * yn) / (g * (2 - g))
    ynp1, n2 = _resout(lambda z: z - second_membre - c*h*f(t + h, z),
                       lambda z: Id - c*h*jac(t + h, z), yg)
    return ynp1, n1 + n2


def integration_BDF2(f, yn, ynm1, t, h, jac):
    # Schéma à deux pas : y_{n+1} - 4/3 y_n + 1/3 y_{n-1} = 2/3 h f(t_{n+1}, y_{n+1})
    Id = np.eye(len(yn))
    second_membre = 4/3*yn - 1/3*ynm1
    return _resout(lambda z: z - second_membre - 2/3*h*f(t + h, z),
                   lambda z: Id - 2/3*h*jac(t + h, z), yn)


# Schémas à pas fixe disponibles : nom -> (un pas, implicite ?)
SCHEMAS = {
    "Euler explicite": (integration_Euler, False),
    "Heun (RK2)": (integration_RK2, False),
    "RK4": (integration_RK4, False),
    "Euler implicite": (integration_Euler_implicite, True),
    "Crank-Nicolson": (integration_Crank_Nicolson, True),
    "Point milieu": (integration_point_milieu, True),
    "BDF2": (integration_BDF2, True),
    "SDIRK2": (integration_SDIRK2, True),
    "TR-BDF2": (integration_TR_BDF2, True),
}


# %% 2) Intégration complète avec un schéma

def integration_temporelle_schema(nom_schema, f, y0, Nt, h, jac=None,
                                  seuil_explosion=1e10):
    # Intégration de dy/dt = f(t, y) sur Nt instants t_n = n*h avec le schéma
    # nom_schema (une clé de SCHEMAS). jac ne sert qu'aux schémas implicites.
    # Renvoie y_res (len(y0), Nt) et un dictionnaire info : nombre total
    # d'appels à f ("nfev") et explosion numérique ("explosion" : True si
    # |y| > seuil_explosion ; la suite vaut alors NaN)
    schema, implicite = SCHEMAS[nom_schema]
    nb_appels = {"Euler explicite": 1, "Heun (RK2)": 2, "RK4": 4}
    y_res = np.full((len(y0), Nt), np.nan)
    y_res[:, 0] = y0
    nfev = 0
    explosion = False
    for ii in range(Nt - 1):
        if nom_schema == "BDF2" and ii == 0:
            # Démarrage de BDF2 (schéma à deux pas) : un pas de SDIRK2
            ynp1, n = integration_SDIRK2(f, y_res[:, 0], 0, h, jac)
        elif nom_schema == "BDF2":
            ynp1, n = schema(f, y_res[:, ii], y_res[:, ii-1], ii*h, h, jac)
        elif implicite:
            ynp1, n = schema(f, y_res[:, ii], ii*h, h, jac)
        else:
            ynp1, n = schema(f, y_res[:, ii], ii*h, h), nb_appels[nom_schema]
        nfev += n
        if not np.all(np.isfinite(ynp1)) or np.max(np.abs(ynp1)) > seuil_explosion:
            explosion = True
            break
        y_res[:, ii+1] = ynp1
    return y_res, {"nfev": nfev, "explosion": explosion}


# %% 3) RK4 pour Lorenz : versions pur Python, NumPy (boucles), NumPy vectorisé, Numba

def integration_temporelle_python(y0, Nt, Pr, r, b, h):
    # Pur Python : trois flottants X, Y, Z, f écrite composante par composante,
    # trajectoire stockée dans trois listes
    X = [0.0] * Nt
    Y = [0.0] * Nt
    Z = [0.0] * Nt
    x, y, z = float(y0[0]), float(y0[1]), float(y0[2])
    X[0], Y[0], Z[0] = x, y, z
    for ii in range(Nt - 1):
        k1x, k1y, k1z = Pr*(y - x), -x*z + r*x - y, x*y - b*z
        x2, y2, z2 = x + h/2*k1x, y + h/2*k1y, z + h/2*k1z
        k2x, k2y, k2z = Pr*(y2 - x2), -x2*z2 + r*x2 - y2, x2*y2 - b*z2
        x3, y3, z3 = x + h/2*k2x, y + h/2*k2y, z + h/2*k2z
        k3x, k3y, k3z = Pr*(y3 - x3), -x3*z3 + r*x3 - y3, x3*y3 - b*z3
        x4, y4, z4 = x + h*k3x, y + h*k3y, z + h*k3z
        k4x, k4y, k4z = Pr*(y4 - x4), -x4*z4 + r*x4 - y4, x4*y4 - b*z4
        x = x + h/6 * (k1x + 2*k2x + 2*k3x + k4x)
        y = y + h/6 * (k1y + 2*k2y + 2*k3y + k4y)
        z = z + h/6 * (k1z + 2*k2z + 2*k3z + k4z)
        X[ii+1], Y[ii+1], Z[ii+1] = x, y, z
    return np.array([X, Y, Z])


def integration_temporelle(y0, Nt, Pr, r, b, h):
    # NumPy à boucles : l'état est un tableau de taille 3, une boucle en temps
    y_res = np.zeros((3, Nt))
    y_res[:, 0] = y0
    for ii in range(Nt - 1):
        y_res[:, ii+1] = integration_RK4(
            lambda t, yn: systeme_dynamique(t, yn, Pr, r, b),
            y_res[:, ii], ii*h, h)
    return y_res


def integration_temporelle_vectorisee(y0, Nt, Pr, rs, b, h):
    # NumPy vectorisé : toutes les valeurs de rs à la fois. L'état est de forme
    # (3, len(rs)) et le broadcasting applique f aux len(rs) systèmes en une fois.
    # Résultat de forme (len(rs), 3, Nt) : y_res_list[ii] pour r = rs[ii]
    rs = np.asarray(rs, dtype=float)
    y_res_list = np.zeros((Nt, 3, rs.size))
    y_res_list[0] = np.reshape(y0, (3, 1))
    for ii in range(Nt - 1):
        y_res_list[ii+1] = integration_RK4(
            lambda t, yn: systeme_dynamique(t, yn, Pr, rs, b),
            y_res_list[ii], ii*h, h)
    return np.transpose(y_res_list, (2, 1, 0))


def _integration_explicite_tableaux(y0, Nt, Pr, r, b, h, schema, f):
    # Euler (schema = 0), Heun (1) ou RK4 (2) pour Lorenz, sans lambda (pour
    # numba) ; mêmes opérations, dans le même ordre, que les fonctions de pas
    y_res = np.zeros((3, Nt))
    y_res[:, 0] = y0
    for ii in range(Nt - 1):
        yn = y_res[:, ii]
        t = ii*h
        k1 = f(t, yn, Pr, r, b)
        if schema == 0:
            y_res[:, ii+1] = yn + h * k1
        elif schema == 1:
            k2 = f(t + h, yn + h*k1, Pr, r, b)
            y_res[:, ii+1] = yn + h/2 * (k1 + k2)
        else:
            k2 = f(t + h/2, yn + h/2*k1, Pr, r, b)
            k3 = f(t + h/2, yn + h/2*k2, Pr, r, b)
            k4 = f(t + h, yn + h*k3, Pr, r, b)
            y_res[:, ii+1] = yn + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_res


if NUMBA_DISPONIBLE:
    # Compilation au PREMIER appel (quelques secondes), puis code machine
    _systeme_dynamique_numba = njit(systeme_dynamique)
    _integration_numba = njit(_integration_explicite_tableaux)


def integration_temporelle_numba(y0, Nt, Pr, r, b, h, schema="RK4"):
    # Version compilée par numba ; schema : "Euler explicite", "Heun (RK2)" ou
    # "RK4". C'est la version à utiliser pour tous les calculs longs.
    # Sans numba : même calcul en NumPy (plus lent), avec un avertissement
    code = {"Euler explicite": 0, "Heun (RK2)": 1, "RK4": 2}[schema]
    y0 = np.asarray(y0, dtype=float)
    if not NUMBA_DISPONIBLE:
        warnings.warn("numba n'est pas installé : calcul en NumPy (plus lent)")
        return _integration_explicite_tableaux(y0, Nt, Pr, r, b, h, code,
                                               systeme_dynamique)
    return _integration_numba(y0, int(Nt), float(Pr), float(r), float(b),
                              float(h), code, _systeme_dynamique_numba)


def integration_temporelle_flottant(y0, Nt, Pr, r, b, h, flottant=np.float32):
    # RK4 en NumPy où TOUS les nombres sont du type flottant (ex. np.float32,
    # simple précision) : sans conversion, NumPy repasserait en double précision
    c = flottant
    y_res = np.zeros((3, Nt), dtype=c)
    y_res[:, 0] = np.asarray(y0, dtype=c)
    Pr, r, b, h = c(Pr), c(r), c(b), c(h)
    deux, six = c(2), c(6)

    def f(y):
        return np.array([Pr*(y[1] - y[0]), -y[0]*y[2] + r*y[0] - y[1],
                         y[0]*y[1] - b*y[2]], dtype=c)
    for ii in range(Nt - 1):
        yn = y_res[:, ii]
        k1 = f(yn)
        k2 = f(yn + h/deux*k1)
        k3 = f(yn + h/deux*k2)
        k4 = f(yn + h*k3)
        y_res[:, ii+1] = yn + h/six * (k1 + deux*k2 + deux*k3 + k4)
    return y_res
