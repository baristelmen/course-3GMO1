"""Fonctions de tracé de la partie II. Seuls les tracés qui demandent un calcul
(pente, zone stable, enveloppe) sont fournis ; les autres figures se tracent
directement avec matplotlib. Chaque fonction trace dans les axes ax donnés,
sinon dans une nouvelle figure, et renvoie la figure."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analyse import fonction_stabilite, taux_enveloppe

# Couleur et style de chaque schéma, identiques sur toutes les figures
# (trait plein : explicite ; tirets : implicite)
STYLES_SCHEMAS = {
    "Euler explicite": dict(color="tab:red", linestyle="-", marker="o"),
    "Heun (RK2)": dict(color="tab:orange", linestyle="-", marker="s"),
    "RK4": dict(color="tab:green", linestyle="-", marker="^"),
    "Euler implicite": dict(color="tab:blue", linestyle="--", marker="o"),
    "Crank-Nicolson": dict(color="tab:purple", linestyle="--", marker="s"),
    "Point milieu": dict(color="tab:pink", linestyle=":", marker="D"),
    "BDF2": dict(color="tab:brown", linestyle="--", marker="v"),
    "SDIRK2": dict(color="tab:cyan", linestyle="--", marker="^"),
    "TR-BDF2": dict(color="tab:olive", linestyle="--", marker="P"),
}
L_STABLE = {"Euler implicite", "BDF2", "SDIRK2", "TR-BDF2"}


def style_schema(nom, marqueur=True):
    # Style d'une courbe (celui du schéma s'il est connu)
    style = dict(STYLES_SCHEMAS.get(nom, dict(marker="o")))
    if not marqueur:
        style.pop("marker", None)
    return style


def trace_convergence(hs, erreurs, ordres=None, hmin=None, hmax=None, ax=None,
                      titre="Convergence en pas de temps", ylabel="Erreur"):
    # Courbes d'erreur en fonction du pas h (log-log), avec la pente mesurée
    # (np.polyfit sur hmin <= h <= hmax) et, si ordres est donné ({nom : p}),
    # la droite de pente théorique C h^p en pointillés. Les zones exclues de la
    # mesure de pente sont grisées. Plusieurs appels sur les mêmes axes sont
    # possibles (une plage de mesure différente par courbe).
    # Renvoie la figure et le dictionnaire {nom : pente mesurée}
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    else:
        fig = ax.get_figure()
    hs = np.asarray(hs, dtype=float)
    pentes = {}
    for nom, erreur in erreurs.items():
        erreur = np.asarray(erreur, dtype=float)
        garde = np.isfinite(erreur) & (erreur > 0)
        if hmin is not None:
            garde &= hs >= hmin
        if hmax is not None:
            garde &= hs <= hmax
        p, logC = np.polyfit(np.log(hs[garde]), np.log(erreur[garde]), 1)
        pentes[nom] = p
        texte = "%s : pente mesurée %.2f" % (nom, p)
        if ordres is not None and nom in ordres:
            texte += " (théorie %d)" % ordres[nom]
        style = style_schema(nom)
        style["linestyle"] = "-"
        (ligne,) = ax.loglog(hs, erreur, linewidth=1.5, markersize=6, label=texte, **style)
        if ordres is not None and nom in ordres:
            i0 = np.where(garde)[0][np.sum(garde) // 2]
            ax.loglog(hs, erreur[i0] * (hs / hs[i0])**ordres[nom], ":",
                      color=ligne.get_color(), linewidth=1.2)
    # Zones exclues de la mesure (grisées) ; on mémorise sur ax quelles
    # entrées de légende sont utiles (pente théorique, zone grisée)
    ax.zones_grisees = getattr(ax, "zones_grisees", False) or hmin is not None or hmax is not None
    ax.pentes_theoriques = getattr(ax, "pentes_theoriques", False) or ordres is not None
    if hmin is not None:
        ax.axvspan(np.min(hs) / 1.5, hmin, color="0.92", zorder=0)
    if hmax is not None:
        ax.axvspan(hmax, np.max(hs) * 1.5, color="0.92", zorder=0)
    ax.set_xlabel("Pas de temps $h$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(titre, fontweight="bold", fontsize=13)
    ax.grid(True, which="major", alpha=0.6)
    ax.grid(True, which="minor", alpha=0.2)
    # Légende à droite, avec une seule entrée pour la pente théorique et la zone grisée
    poignees, noms = ax.get_legend_handles_labels()
    gardes = [(p, n) for p, n in zip(poignees, noms) if not n.startswith("_")]
    poignees, noms = [p for p, _ in gardes], [n for _, n in gardes]
    extra = []
    if ax.pentes_theoriques:
        extra.append((plt.Line2D([], [], linestyle=":", color="k"), "pente théorique $C h^p$"))
    if ax.zones_grisees:
        extra.append((Patch(color="0.92"), "zone exclue de la mesure de pente\n"
                                           "(régime pré-asymptotique ou plancher d'erreur)"))
    for poignee, nom in extra:
        if nom not in noms:
            poignees.append(poignee)
            noms.append(nom)
    ax.legend(poignees, noms, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    return fig, pentes


def trace_domaines_stabilite(noms_schemas, xlim=(-6, 3), ylim=(-4, 4)):
    # Un panneau par schéma : zone STABLE {z : |R(z)| <= 1} en bleu, frontière
    # |R(z)| = 1, axe imaginaire en noir épais ; test de la A-stabilité (le
    # demi-plan Re z <= 0 est-il entièrement bleu ?) indiqué dans le titre
    n = len(noms_schemas)
    ncol = min(3, n)
    nlig = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nlig, ncol, figsize=(3.6*ncol, 3.5*nlig),
                            layout="constrained", squeeze=False)
    xg, yg = np.meshgrid(np.linspace(*xlim, 451), np.linspace(*ylim, 401))
    with np.errstate(divide="ignore", invalid="ignore"):
        for ax, nom in zip(axs.flat, noms_schemas):
            module = np.abs(fonction_stabilite(nom, xg + 1j*yg))
            # tolérance 1e-9 : sur l'axe imaginaire, |R| = 1 exactement pour
            # Crank-Nicolson et le point milieu, mais 1 + 1e-16 en flottants
            stable = np.where(np.isfinite(module), module <= 1 + 1e-9, False)
            ax.contourf(xg, yg, stable.astype(float), levels=[0.5, 1.5], colors=["#9ecae1"])
            ax.contour(xg, yg, module, levels=[1], colors="#08519c", linewidths=1.5)
            ax.axvline(0, color="black", linewidth=2)
            ax.axhline(0, color="black", linewidth=0.5)
            a_stable = np.all(stable[xg <= 0])
            ax.set_title("%s\nA-stable : %s" % (nom, "oui" if a_stable else "non"),
                         fontweight="bold", fontsize=12)
            ax.set_aspect("equal")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.4)
    for ax in axs.flat[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel(r"Re $z$ ($z = h\lambda$)", fontsize=11)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Im $z$", fontsize=11)
    fig.suptitle("Domaines de stabilité : en bleu, $|R(z)| \\leq 1$ "
                 "(le schéma est stable si $h\\lambda$ est dans la zone bleue)",
                 fontweight="bold", fontsize=13)
    return fig


def trace_facteurs_amplification(noms_schemas, ymax=4, xmax=1e6):
    # Gauche : |R(iy)|, facteur par pas d'une oscillation non amortie
    # (> 1 : amplifiée ; = 1 : conservée ; < 1 : amortie).
    # Droite : |R(-x)|, facteur par pas d'un mode amorti ; L-stable si -> 0
    # quand x = -h lambda -> +infini
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    y = np.linspace(0, ymax, 400)
    x = np.logspace(-2, np.log10(xmax), 400)
    # Fonctions de stabilité identiques : point milieu = Crank-Nicolson,
    # TR-BDF2 = SDIRK2 (courbes superposées)
    identiques = {"Point milieu": "Crank-Nicolson", "TR-BDF2": "SDIRK2"}
    for nom in noms_schemas:
        style = style_schema(nom, marqueur=False)
        etiquette = nom
        if nom in identiques and identiques[nom] in noms_schemas:
            etiquette += " (= %s)" % identiques[nom]
        axs[0].plot(y, np.abs(fonction_stabilite(nom, 1j*y)), linewidth=2, label=etiquette,
                    **style)
        axs[1].semilogx(x, np.abs(fonction_stabilite(nom, -x)), linewidth=2,
                        label="%s (%s)" % (etiquette, "L-stable" if nom in L_STABLE
                                                 else "non L-stable"), **style)
    for ax in axs:
        ax.axhline(1, color="black", linewidth=1.5)
        ax.set_ylim(0, 1.6)
        ax.grid(True, alpha=0.5)
    axs[0].fill_between(y, 1, 1.6, color="tab:red", alpha=0.08)
    axs[0].text(0.05, 1.5, "> 1 : amplifie (injecte de l'énergie)", fontsize=10,
                color="tab:red")
    axs[0].text(ymax*0.62, 1.04, "= 1 : conserve l'énergie", fontsize=10)
    axs[0].text(ymax*0.62, 0.04, "< 1 : amortit (dissipe de l'énergie)", fontsize=10,
                color="tab:blue")
    axs[0].set_title("Oscillation non amortie ($\\lambda = i\\omega$) : $|R(iy)|$",
                     fontweight="bold", fontsize=13)
    axs[0].set_xlabel(r"$y = h\omega$", fontsize=12)
    axs[0].set_ylabel(r"$|R(iy)|$", fontsize=12)
    axs[0].legend(fontsize=9, loc="lower left")
    axs[1].set_title("Mode amorti ($\\lambda < 0$) : $|R(-x)|$\n"
                     "L-stable si $|R(-x)| \\to 0$ quand $h\\lambda \\to -\\infty$",
                     fontweight="bold", fontsize=13)
    axs[1].set_xlabel(r"$x = -h\lambda$ (échelle log)", fontsize=12)
    axs[1].set_ylabel(r"$|R(-x)|$", fontsize=12)
    axs[1].legend(fontsize=9, loc="upper left")
    return fig


def trace_enveloppe(signal, ts, ax=None, titre="", nom="signal", tmin=None,
                    tmax=None, sigma_predit=None):
    # Signal oscillant (gris), maxima successifs de |signal| (rouge), enveloppe
    # exponentielle ajustée (tirets) ; le taux mesuré sigma (et le taux prédit)
    # est écrit dans le titre. Renvoie la figure et sigma
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    else:
        fig = ax.get_figure()
    sigma, maxima, ts_max, coefficients = taux_enveloppe(signal, ts, tmin, tmax)
    ax.plot(ts, np.abs(signal), linewidth=0.5, color="0.75", label="|%s|" % nom)
    ax.plot(ts_max, maxima, "o", color="tab:red", markersize=3, label="maxima successifs")
    if coefficients is not None:
        ax.plot(ts_max, np.exp(np.polyval(coefficients, ts_max)), "--", color="black",
                linewidth=2, label="enveloppe ajustée")
    texte = titre + "\n" + r"$\sigma$ mesuré = %+.3e" % sigma
    if sigma_predit is not None:
        texte += r", prédit = %+.3e" % sigma_predit
    ax.set_yscale("log")
    if len(maxima) > 0:
        ax.set_ylim(np.min(maxima) / 30, np.max(maxima) * 3)
    ax.set_title(texte, fontweight="bold", fontsize=11)
    ax.set_xlabel("Temps $t$ (s)")
    ax.set_ylabel("Amplitude (échelle log)")
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")
    return fig, sigma
