# Convection de Rayleigh-Bénard — partie II (bonus)

Le travail se fait dans le notebook **`partie2_etudiant.ipynb`**, qui suit
l'énoncé : les explications et les questions alternent, et chaque question est
suivie d'une cellule « À compléter ».

## Installation (détails dans l'énoncé, section 1)

Bibliothèques : numpy, scipy, matplotlib, numba (conseillé), scikit-learn
(section 7), ipykernel et notebook (notebook). Dans un terminal ouvert dans ce dossier :

* **Windows + VSCode** (extensions *Python* et *Jupyter*) : `py -m pip install -r requirements.txt`
* **macOS** : `python3 -m pip install -r requirements.txt`
* **Linux** : `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`

Puis `python verifie_installation.py` (ou `py` sous Windows).

## Contenu

| Fichier | Rôle |
|---|---|
| `lorenz.py` | système de Lorenz, jacobienne, points fixes, système sans dissipation, oscillateur harmonique |
| `integrateurs.py` | un pas de chaque schéma à pas fixe ; intégration avec un schéma ; RK4 pur Python, NumPy à boucles, NumPy vectorisé, Numba ; simple précision |
| `analyse.py` | références DOP853 (`.json`), maxima précis, corrélation, stabilité des schémas, seuils, enveloppes, balayage quasi statique |
| `traces_partie2.py` | tracés : convergence (pente mesurée), domaines de stabilité, facteurs d'amplification, enveloppe |
| `donnees_apprentissage.py` | données de la section 7 |
| `main_calcul_des_donnees.py` | recalcul des fichiers de `data/` (inutile : ils sont fournis) |
| `tableau_recapitulatif.csv` | tableau de l'annexe A, à compléter (s'ouvre avec un tableur) |
| `verifie_installation.py` | vérification de l'installation |
| `data/` | `reference_r10.json`, `reference_r10_precise.json`, `reference_r28.json`, `amplitudes.json`, `bassins_r10.json`, `bassins_r20.json` |

Conventions : une trajectoire `y_res` est un tableau `(3, Nt)` (lignes X, Y, Z ;
colonne n à l'instant t_n = n h), avec `h = tmax / (Nt-1)`.

## Remarques
* Ouvrir **le dossier** dans VSCode (et non le seul fichier) pour que les
  modules soient trouvés par `import`.
* Exécuter les cellules du notebook dans l'ordre. Après une modification d'un
  fichier `.py`, redémarrer le noyau.
* Avec numba, aucune cellule ne dure plus d'une minute sur un ordinateur modeste.
