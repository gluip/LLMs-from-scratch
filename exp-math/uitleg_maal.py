# Hoe vermenigvuldigt dit netwerk? De onderbouwing van verslag-maal.html.
#
# Anders dan bij optellen is het model hier niet samen te vouwen tot één
# formule: er zit een ReLU in en die is niet lineair. Wat wel kan is per
# stadium meten wat er lineair afleesbaar is uit de interne toestand.
#
# Draaien:  .venv/bin/python -u exp-math/uitleg_maal.py   (~4 min)

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
BESTE = dict(n_embed=32, positie=True, leer_aandacht=True, uit_proj=True,
             n_koppen=8, ff=True)
STAPPEN = 30000


def model_en_toestanden(seed=0):
    """Traint het model en geeft de interne toestanden op de '='-positie terug.

    h1 = na de aandacht, vóór de feedforward
    h2 = na de feedforward, vlak voor de uitlees
    """
    tok, tw, xtr, ytr, xte, yte = r.laad("drie")
    m, _, acc = r.train(xtr, ytr, xte, yte, tok.vocab_size, seed=seed,
                        n_stappen=STAPPEN, **BESTE)
    m.eval()
    paren = [(a, b) for a in range(10) for b in range(10)]
    x = torch.tensor([tok.encode([str(a), "*", str(b), "="]) for a, b in paren])
    with torch.no_grad():
        h = m.embed(x) + m.pos(torch.arange(4))
        q, k, vs = m._splits(m.Q(h)), m._splits(m.K(h)), m._splits(m.V(h))
        rauw = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(m.kop_dim), -1)
        meng = (rauw @ vs).transpose(1, 2).contiguous().view(100, 4, -1)
        h1 = h + m.W_o(meng)
        h2 = h1 + m.ff(h1)
        # ter vergelijking: dezelfde som met de aandacht bevroren op het
        # gemiddelde, zodat hij niet meer van de inhoud afhangt
        vast = rauw.mean(0, keepdim=True).expand(100, -1, -1, -1)
        meng_v = (vast @ vs).transpose(1, 2).contiguous().view(100, 4, -1)
        h1_vast = h + m.W_o(meng_v)
    a = torch.tensor([float(p[0]) for p in paren])
    b = torch.tensor([float(p[1]) for p in paren])
    return m, acc, h1[:, -1], h2[:, -1], h1_vast[:, -1], a, b


def voorspel_kruisgevalideerd(H, doel, vouwen=5, seed=0):
    """Lees `doel` uit toestand H met een lineaire probe, en geef per punt de
    voorspelling die gemaakt is door een probe die dat punt NIET zag.

    Dat laatste is essentieel: 32 dimensies plus een bias zijn 33 vrije
    parameters voor 100 punten, en daarmee fit je ook pure ruis nog aardig.
    Meten op de eigen trainingsdata meet de probe, niet het model.
    """
    g = torch.Generator().manual_seed(seed)
    orde = torch.randperm(len(H), generator=g)
    voorspeld = torch.zeros(len(H))
    for v in range(vouwen):
        test = orde[v::vouwen]
        train = torch.tensor([i for i in orde.tolist() if i not in set(test.tolist())])
        X = torch.cat([H[train], torch.ones(len(train), 1)], 1)
        opl = torch.linalg.lstsq(X, doel[train].unsqueeze(1)).solution
        Xt = torch.cat([H[test], torch.ones(len(test), 1)], 1)
        voorspeld[test] = (Xt @ opl).squeeze(1)
    return voorspeld


def r2_van(doel, voorspeld):
    return (1 - ((doel - voorspeld) ** 2).sum() / ((doel - doel.mean()) ** 2).sum()).item()


def figuur_verstrooiing(h1, h2, doel, bestand):
    """Voorspeld tegen werkelijk, vóór en ná de feedforward.

    De band van een halve eenheid om de diagonaal is wat telt: daarbuiten
    rondt het antwoord naar het verkeerde gehele getal af.
    """
    fig, assen = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (H, naam) in zip(assen, ((h1, "vóór de feedforward"),
                                     (h2, "ná de feedforward"))):
        p = voorspel_kruisgevalideerd(H, doel)
        fout = (doel - p)
        goed = (p.round() == doel)
        ax.fill_between([-4, 85], [-4.5, 84.5], [-3.5, 85.5],
                        color="#2f7d5f", alpha=0.18, label="binnen een halve eenheid")
        ax.plot([-4, 85], [-4, 85], "-", color="#7b8894", lw=1)
        ax.scatter(doel[goed], p[goed], s=26, color="#2a6f97", label="rondt goed af")
        ax.scatter(doel[~goed], p[~goed], s=26, color="#b0574f", label="rondt fout af")
        ax.set_xlabel("het werkelijke product a · b")
        ax.set_title(f"{naam}\ntypische fout {fout.std():.2f} — "
                     f"{goed.float().mean():.0%} goed", fontsize=11)
        ax.set_xlim(-4, 85); ax.set_ylim(-8, 88); ax.grid(alpha=0.25)
    assen[0].set_ylabel("wat er uit de toestand te lezen is")
    assen[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Dezelfde stap in R², een wereld van verschil in het antwoord",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_bijdragen(h1, h2, h1_vast, doel, bestand):
    """De drie bijdragen aan het product, in R² en in typische fout."""
    namen = ["alleen lineair\n(aandacht bevroren)", "+ inhoudsafhankelijke\naandacht",
             "+ de feedforward"]
    fouten, r2s = [], []
    for H in (h1_vast, h1, h2):
        p = voorspel_kruisgevalideerd(H, doel)
        fouten.append((doel - p).std().item()); r2s.append(r2_van(doel, p))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.bar(namen, r2s, 0.55, color=["#b8c4cc", "#7ba8c4", "#2a6f97"])
    for i, v in enumerate(r2s):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.7, 1.04); ax1.set_ylabel("R²  (hoe goed afleesbaar)")
    ax1.set_title("In R² lijkt de laatste stap klein", fontsize=11)
    ax1.grid(alpha=0.25, axis="y")

    ax2.bar(namen, fouten, 0.55, color=["#b8c4cc", "#7ba8c4", "#2a6f97"])
    for i, v in enumerate(fouten):
        ax2.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.axhline(0.5, color="#b0574f", ls="--", lw=1.4)
    ax2.text(0.05, 1.35, "0,5 — hierboven rondt het antwoord fout af",
             color="#b0574f", fontsize=9, ha="left")
    ax2.set_ylabel("typische fout in het antwoord")
    ax2.set_title("In het antwoord is hij beslissend", fontsize=11)
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)
    return r2s, fouten


def toon_probe_val():
    """Waarom een probe kruisgevalideerd moet worden: doe hem op pure ruis."""
    torch.manual_seed(0)
    doel = torch.randn(100)
    alles, helft, rest = torch.arange(100), torch.arange(50), torch.arange(50, 100)
    print("\nde probe-val, gedemonstreerd op PURE RUIS (niets te vinden):")
    print(f"  {'dimensies':>10}{'R² op eigen data':>20}{'R² op nieuwe data':>20}")
    for d in (2, 8, 32, 99):
        H = torch.randn(100, d)
        def fit(idx):
            X = torch.cat([H[idx], torch.ones(len(idx), 1)], 1)
            return torch.linalg.lstsq(X, doel[idx].unsqueeze(1)).solution
        def r2(opl, idx):
            X = torch.cat([H[idx], torch.ones(len(idx), 1)], 1)
            res = doel[idx] - (X @ opl).squeeze(1)
            return 1 - (res ** 2).sum() / ((doel[idx] - doel[idx].mean()) ** 2).sum()
        print(f"  {d:>10}{r2(fit(alles), alles):>20.3f}{r2(fit(helft), rest):>20.3f}")


if __name__ == "__main__":
    FIGUREN.mkdir(exist_ok=True)
    print("model trainen (30.000 stappen, even geduld)...")
    m, acc, h1, h2, h1_vast, a, b = model_en_toestanden()
    doel = a * b
    print(f"test-accuratesse: {acc:.0%}\n")

    print("wat is er lineair afleesbaar op de '='-positie? (kruisgevalideerd)")
    print(f"  {'doel':>10}{'vóór ff':>10}{'ná ff':>10}")
    for naam, d in (("a", a), ("b", b), ("a+b", a + b), ("a*b", doel),
                    ("ruis", torch.randn(100))):
        print(f"  {naam:>10}{r2_van(d, voorspel_kruisgevalideerd(h1, d)):>10.3f}"
              f"{r2_van(d, voorspel_kruisgevalideerd(h2, d)):>10.3f}")

    toon_probe_val()

    print("\nfiguren maken...")
    figuur_verstrooiing(h1, h2, doel, FIGUREN / "maal_verstrooiing.png")
    r2s, fouten = figuur_bijdragen(h1, h2, h1_vast, doel, FIGUREN / "maal_bijdragen.png")
    print(f"  R²:            {[round(v,4) for v in r2s]}")
    print(f"  typische fout: {[round(v,2) for v in fouten]}")
    print(f"geschreven naar {FIGUREN}")
