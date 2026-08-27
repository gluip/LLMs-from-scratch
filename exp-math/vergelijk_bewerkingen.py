# Optellen versus aftrekken: wat verandert er als de bewerking niet meer
# commutatief is?
#
# Bij optellen bleek de aandacht een vaste middelingsstap: uniform, en
# onafhankelijk van de cijfers (zie EXPERIMENTEN.md, experiment 5). Aftrekken
# is niet commutatief, dus daar MOET het model a en b uit elkaar houden. Dit
# script laat zien hoe het dat doet, en wat het kost.
#
# Draaien:  .venv/bin/python -u exp-math/vergelijk_bewerkingen.py

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rekenen as r

FIGUREN = Path(__file__).parent / "figuren"

# de configuratie waarmee allebei de bewerkingen foutloos lukken, zodat het
# verschil in gedrag niet aan een verschil in architectuur kan liggen
EERLIJK = dict(n_embed=16, positie=True, leer_aandacht=True, uit_proj=True)


def getraind(bewerking, seed=0, **knoppen):
    tok, tw, xtr, ytr, xte, yte = r.laad(bewerking)
    model, a_tr, a_te = r.train(xtr, ytr, xte, yte, tok.vocab_size, seed=seed, **knoppen)
    return model, tok, a_te


@torch.no_grad()
def aandacht_kaart(model, tok, teken):
    """Aandachtsgewicht vanaf '=' naar positie a, voor alle 100 sommen.

    Geeft een 10x10-kaart terug: rij = a, kolom = b. Bij optellen is die vlak,
    bij aftrekken loopt er een duidelijke helling in.
    """
    x = torch.tensor([tok.encode([str(a), teken, str(b), "="])
                      for a in range(10) for b in range(10)])
    h = model.embed(x)
    if model.pos is not None:
        h = h + model.pos(torch.arange(4))
    q, k = model._splits(model.Q(h)), model._splits(model.K(h))
    g = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(model.kop_dim), -1)
    return g[:, 0, 3, 0].view(10, 10)      # vanaf '=', kop 0, naar positie a


def figuur_aandacht_kaart(bestand):
    fig, assen = plt.subplots(1, 2, figsize=(11, 4.5))
    kaarten = []
    for bewerking in ("optellen", "aftrekken"):
        model, tok, acc = getraind(bewerking, **EERLIJK)
        kaarten.append((bewerking, aandacht_kaart(model, tok, r.BEWERKINGEN[bewerking][1]), acc))

    # gemeenschappelijke kleurschaal, anders lijkt vlakke ruis net zo dramatisch
    laag = min(k.min() for _, k, _ in kaarten)
    hoog = max(k.max() for _, k, _ in kaarten)
    for ax, (bewerking, kaart, acc) in zip(assen, kaarten):
        beeld = ax.imshow(kaart, cmap="RdBu_r", vmin=laag, vmax=hoog, origin="lower")
        ax.set_xlabel("b"); ax.set_ylabel("a")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_title(f"{bewerking}   (test {acc:.0%})\nspreiding {kaart.std():.4f}", fontsize=11)
        fig.colorbar(beeld, ax=ax, fraction=0.046)
    fig.suptitle("Hoeveel aandacht geeft de '='-positie aan de eerste operand?",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


LADDER = [
    ("kaal", dict()),
    ("+ positie", dict(positie=True)),
    ("+ aandacht", dict(positie=True, leer_aandacht=True)),
    ("+ W_o", dict(positie=True, leer_aandacht=True, uit_proj=True)),
    ("+ feedforward", dict(positie=True, leer_aandacht=True, uit_proj=True, ff=True)),
]


def figuur_ladder(bestand):
    """Welke onderdelen heeft elke bewerking nodig? Opbouwend van kaal naar vol."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    breedte = 0.36
    for i, (bewerking, kleur) in enumerate((("optellen", "#2a6f97"), ("aftrekken", "#b0574f"))):
        gems, mins = [], []
        for naam, knoppen in LADDER:
            _, te, _ = r.meet(bewerking, n_embed=16, **knoppen)
            gems.append(te.mean().item() * 100); mins.append(te.min().item() * 100)
            print(f"  {bewerking:>10} {naam:>16}: {te.mean():.0%} (min {te.min():.0%})")
        x = torch.arange(len(LADDER)) + (i - 0.5) * breedte
        ax.bar(x, gems, breedte, color=kleur, label=bewerking)
        # de ondergrens als streepje: daar zit het echte verhaal
        ax.scatter(x, mins, color="#1b232c", s=16, zorder=3,
                   label="laagste seed" if i == 0 else None)
    ax.set_xticks(range(len(LADDER)), [n for n, _ in LADDER])
    ax.set_ylabel("goed van de 20 achtergehouden sommen (%)")
    ax.set_ylim(0, 118); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Optellen kan alles missen; aftrekken heeft W_o nodig", fontsize=12)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_bodem(bestand):
    """Hoe klein kan het model per bewerking?"""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    opzet = {"optellen": dict(), "aftrekken": dict(positie=True, leer_aandacht=True, uit_proj=True)}
    for bewerking, kleur, stijl in (("optellen", "#2a6f97", "o-"), ("aftrekken", "#b0574f", "s--")):
        maten, accs, pars = [], [], []
        for ne in (2, 4, 8, 16):
            _, te, n = r.meet(bewerking, n_embed=ne, **opzet[bewerking])
            maten.append(ne); accs.append(te.min().item() * 100); pars.append(n)
            print(f"  {bewerking:>10} n_embed={ne:>2}: min {te.min():.0%}, {n} parameters")
        ax.plot(pars, accs, stijl, color=kleur, lw=1.8, ms=8, label=bewerking)
        for p, a, ne in zip(pars, accs, maten):
            ax.annotate(f"{ne}", (p, a), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=kleur)
    ax.set_xscale("log"); ax.set_xlabel("parameters (log)")
    ax.set_ylabel("laagste seed, % goed"); ax.set_ylim(0, 112)
    ax.axhline(100, color="#2f7d5f", ls=":", lw=1.2)
    ax.set_title("De prijs van niet-commutatief zijn\n(labels = n_embed)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_getekend(bestand):
    """De ruil: softmax-gewichten [0,1] tegenover getekende gewichten [-1,1].

    Softmax dwingt een convexe combinatie af - de uitvoer is altijd een gewogen
    GEMIDDELDE van de value-vectoren en ligt dus binnen hun omhullende. Een
    verschil ligt daarbuiten. Met 2*softmax-1 mogen er minnen in, en dan is
    v(a) - v(b) direct uit te drukken. Dat helpt aftrekken en schaadt optellen.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    breedte = 0.36
    posities = torch.arange(2)
    for i, (getekend, naam, kleur) in enumerate(
            ((False, "softmax  [0, 1]", "#2a6f97"), (True, "2·softmax − 1  [−1, 1]", "#8a6bb0"))):
        gems, mins = [], []
        for bewerking in ("optellen", "aftrekken"):
            _, te, _ = r.meet(bewerking, seeds=range(10), n_embed=16,
                              positie=True, leer_aandacht=True, getekend=getekend)
            gems.append(te.mean().item() * 100); mins.append(te.min().item() * 100)
            print(f"  {bewerking:>10} {naam:>24}: {te.mean():.0%} (min {te.min():.0%})")
        x = posities + (i - 0.5) * breedte
        ax.bar(x, gems, breedte, color=kleur, label=naam)
        ax.scatter(x, mins, color="#1b232c", s=18, zorder=3,
                   label="laagste seed" if i == 0 else None)
    ax.set_xticks(posities.tolist(), ["optellen", "aftrekken"])
    ax.set_ylabel("goed van de 20 achtergehouden sommen (%)")
    ax.set_ylim(0, 118); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Negatieve gewichten toestaan is een ruil, geen verbetering",
                 fontsize=12)
    ax.legend(loc="upper center", ncol=3, fontsize=8.5)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)



SMAKEN = [
    ("softmax", "softmax\n[0, 1]", "#2a6f97"),
    ("getekend", "2·g − 1\n[−1, 1]", "#8a6bb0"),
    ("gecentreerd", "g − gem(g)\nsom 0", "#b0574f"),
    ("verschil", "sm₁ − sm₂\nsom 0", "#c08a2e"),
    ("tanh", "tanh(aff)\nvrij", "#2f7d5f"),
]


def figuur_smaken(bestand):
    """Vijf manieren om van affiniteiten naar gewichten te gaan.

    Softmax kan niet aftrekken, 2*g-1 kan niet meer goed optellen, gecentreerd
    heeft een negatieve kant die met 1/T verzwakt, verschil kost een extra Q/K,
    en tanh doet allebei de bewerkingen foutloos zonder extra parameters.
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    breedte = 0.16
    posities = torch.arange(2)
    for i, (soort, label, kleur) in enumerate(SMAKEN):
        gems, mins = [], []
        for bewerking in ("optellen", "aftrekken"):
            _, te, n = r.meet(bewerking, seeds=range(10), n_embed=16,
                              positie=True, leer_aandacht=True, soort=soort)
            gems.append(te.mean().item() * 100); mins.append(te.min().item() * 100)
            print(f"  {bewerking:>10} {soort:>12}: {te.mean():.0%} (min {te.min():.0%}), {n} par")
        x = posities + (i - 2) * breedte
        ax.bar(x, gems, breedte, color=kleur, label=label)
        ax.scatter(x, mins, color="#1b232c", s=13, zorder=3,
                   label="laagste seed" if i == 0 else None)
    ax.set_xticks(posities.tolist(), ["optellen", "aftrekken"], fontsize=11)
    ax.set_ylabel("goed van de 20 achtergehouden sommen (%)")
    ax.set_ylim(0, 132); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Alleen tanh doet allebei de bewerkingen foutloos", fontsize=12.5)
    ax.legend(loc="upper center", ncol=6, fontsize=7.5, columnspacing=1.0)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_koppen(bestand):
    """Helpt multi-head aftrekken? (correctie op experiment 7)"""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    kopjes = [1, 2, 4, 8]
    for wo, naam, kleur, stijl in ((False, "zonder W_o", "#b0574f", "s--"),
                                   (True, "met W_o", "#2a6f97", "o-")):
        gems, mins = [], []
        for K in kopjes:
            _, te, _ = r.meet("aftrekken", seeds=range(10), n_embed=16, positie=True,
                              leer_aandacht=True, n_koppen=K, uit_proj=wo)
            gems.append(te.mean().item() * 100); mins.append(te.min().item() * 100)
            print(f"  aftrekken {K} koppen, W_o={wo}: {te.mean():.0%} (min {te.min():.0%})")
        ax.plot(kopjes, gems, stijl, color=kleur, lw=1.8, ms=8, label=naam)
        ax.plot(kopjes, mins, stijl, color=kleur, lw=1, ms=4, alpha=0.45,
                label=f"{naam}, laagste seed")
    ax.set_xscale("log", base=2); ax.set_xticks(kopjes, [str(k) for k in kopjes])
    ax.set_xlabel("aantal koppen"); ax.set_ylabel("goed van de 20 sommen (%)")
    ax.set_ylim(0, 118); ax.grid(alpha=0.3)
    ax.set_title("Multi-head helpt aftrekken wél — correctie op experiment 7", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)



def figuur_schakelaar(bestand):
    """Hoe een model dat beide bewerkingen kent, omschakelt op het teken.

    Bij de gecombineerde taak moet het model het operator-token lezen en zijn
    gedrag daarop aanpassen. Dat blijkt te gebeuren via de concurrentie die
    softmax afdwingt: aandacht die naar het teken gaat, gaat af van b.
    """
    tok, tw, xtr, ytr, xte, yte = r.laad("beide")
    m, _, acc = r.train(xtr, ytr, xte, yte, tok.vocab_size, seed=0, n_embed=16,
                        positie=True, leer_aandacht=True, n_koppen=2, uit_proj=True)
    m.eval()
    verdeling = {}
    for teken in ("+", "-"):
        x = torch.tensor([tok.encode([str(a), teken, str(b), "="])
                          for a in range(10) for b in range(10)])
        with torch.no_grad():
            h = m.embed(x) + m.pos(torch.arange(4))
            q, k = m._splits(m.Q(h)), m._splits(m.K(h))
            g = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(m.kop_dim), -1)
        verdeling[teken] = g[:, :, 3, :].mean(0)

    labels = ["a", "teken", "b", "="]
    kleuren = ["#2a6f97", "#c08a2e", "#b0574f", "#b8c4cc"]
    fig, assen = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, kop in zip(assen, range(2)):
        onder = torch.zeros(2)
        for j, (lbl, kleur) in enumerate(zip(labels, kleuren)):
            hoogte = torch.tensor([verdeling["+"][kop][j], verdeling["-"][kop][j]])
            ax.bar(["optellen", "aftrekken"], hoogte, 0.55, bottom=onder,
                   color=kleur, label=lbl if kop == 0 else None)
            for i, (h_, o_) in enumerate(zip(hoogte, onder)):
                if h_ > 0.08:
                    ax.text(i, o_ + h_ / 2, f"{lbl}\n{h_:.2f}", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            onder = onder + hoogte
        rol = "leest a" if kop == 0 else "leest b — en schakelt om"
        ax.set_title(f"kop {kop}: {rol}", fontsize=11)
        ax.set_ylim(0, 1)
    assen[0].set_ylabel("aandacht vanaf de '='-positie")
    fig.suptitle(f"Aandacht die naar het teken gaat, gaat af van b  (test {acc:.0%})",
                 fontsize=12.5, y=1.03)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    FIGUREN.mkdir(exist_ok=True)

    print("bewijs dat het kale model aftrekken niet KAN:")
    for bewerking in ("optellen", "aftrekken"):
        model, tok, acc = getraind(bewerking, n_embed=16, positie=True)
        teken = r.BEWERKINGEN[bewerking][1]
        with torch.no_grad():
            heen = torch.tensor([tok.encode([str(a), teken, str(b), "="])
                                 for a in range(10) for b in range(10)])
            terug = torch.tensor([tok.encode([str(b), teken, str(a), "="])
                                  for a in range(10) for b in range(10)])
            verschil = (model(heen) - model(terug)).abs().max()
        print(f"  {bewerking:>10}: test {acc:>4.0%}   max |f(a,b) - f(b,a)| = {verschil:.6f}")
    print("  (0 = het model geeft gegarandeerd hetzelfde antwoord op a-b en b-a)")

    print("\nladder:")
    figuur_ladder(FIGUREN / "ladder.png")
    print("\nbodem:")
    figuur_bodem(FIGUREN / "bodem.png")
    print("\naandachtskaart:")
    figuur_aandacht_kaart(FIGUREN / "aandacht_kaart.png")
    print("\ngetekende aandacht:")
    figuur_getekend(FIGUREN / "getekend.png")
    print("\nvijf smaken:")
    figuur_smaken(FIGUREN / "smaken.png")
    print("\nkoppen:")
    figuur_koppen(FIGUREN / "koppen.png")
    print("\nschakelaar:")
    figuur_schakelaar(FIGUREN / "schakelaar.png")
    print(f"\nfiguren geschreven naar {FIGUREN}")
