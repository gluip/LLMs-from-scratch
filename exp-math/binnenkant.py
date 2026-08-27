# Wat doet de attention-laag eigenlijk? Kijken naar Q, K en V van het '='-teken.
#
# Het model is klein genoeg om helemaal uit te rekenen: een laag, vier koppen,
# vier posities. Deze analyse laat zien dat het optellen niet in de aandacht
# zit maar in de value-vectoren, en dat de aandacht zelf niets doet.
#
# Draaien:  .venv/bin/python -u exp-math/binnenkant.py

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
import wiskunde as w

FIGUREN = Path(__file__).parent / "figuren"
POSITIES = ["a", "+", "b", "="]


def leg_klaar(seed=0, **knoppen):
    """Traint een model en geeft alles terug wat je nodig hebt om erin te kijken."""
    regels = w.laad_sommen()
    tok = w.SomTokenizer(regels)
    ids = torch.tensor([tok.encode(r) for r in regels], dtype=torch.long)
    x, y = ids[:, :w.LENGTE], ids[:, w.LENGTE]
    xtr, ytr, xte, yte = w.splits_train_test(x, y)
    tw = tok.waarden()
    model, _ = w.train(xtr, ytr, tw[ytr], xte, yte, tw[yte], tok, tw,
                       seed=seed, stil=True, **knoppen)
    model.eval()
    return model, tok, x, tw


def invoer_van_de_laag(model, x):
    """h zoals de attention-laag hem binnenkrijgt, inclusief eventuele layernorm."""
    h = model.embed(x) + model.pos_embed(torch.arange(x.shape[1]))
    blok = model.lagen[0]
    return blok.ln1(h) if blok.ln1 is not None else h


@torch.no_grad()
def aandacht(model, x):
    """Aandachtsgewichten per kop: (sommen, koppen, van, naar)."""
    laag = model.lagen[0].attentie
    h = invoer_van_de_laag(model, x)
    q, k = laag._splits(laag.Q(h)), laag._splits(laag.K(h))
    aff = q @ k.transpose(-2, -1) / laag.kop_dim ** 0.5
    aff = aff.masked_fill(torch.triu(torch.ones(4, 4, dtype=torch.bool), 1), float("-inf"))
    return torch.softmax(aff, -1), aff


@torch.no_grad()
def value_van_cijfers(model):
    """De value-vectoren van 0..9, geprojecteerd op hun eigen hoofdas.

    Dit is waar het getal zit: als deze projectie recht evenredig oploopt met
    de cijferwaarde, dan is 'de helft van v(a) plus de helft van v(b)' een
    getal dat evenredig is met a+b - en dat is precies wat uniforme aandacht
    uitrekent.
    """
    laag = model.lagen[0].attentie
    v = laag.V(model.embed.weight[:10] + model.pos_embed.weight[0])
    v = v - v.mean(0)
    _, S, Vh = torch.linalg.svd(v, full_matrices=False)
    return v @ Vh.T[:, 0], (S ** 2 / (S ** 2).sum())[0]


def figuur_aandacht(model, x, bestand):
    """De vier koppen als warmtekaart, plus de affiniteiten voor de softmax."""
    gew, aff = aandacht(model, x)
    fig, assen = plt.subplots(1, 5, figsize=(15, 3.4),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]})
    for kop in range(4):
        ax = assen[kop]
        beeld = gew[:, kop].mean(0)
        ax.imshow(beeld, cmap="Blues", vmin=0, vmax=0.5)
        for i in range(4):
            for j in range(i + 1):
                ax.text(j, i, f"{beeld[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5, color="#123" if beeld[i, j] < 0.35 else "white")
        ax.set_xticks(range(4), POSITIES); ax.set_yticks(range(4), POSITIES)
        ax.set_title(f"kop {kop}", fontsize=10)
        if kop == 0:
            ax.set_ylabel("vanaf positie")
        ax.set_xlabel("kijkt naar")

    # de spreiding: varieert de aandacht uberhaupt met de som?
    ax = assen[4]
    rij = gew[:, :, 3, :]
    ax.bar(range(4), rij.mean((0, 1)), yerr=rij.std(0).mean(0) * 50,
           color="#2a6f97", capsize=4)
    ax.axhline(0.25, color="#b0574f", ls="--", lw=1.2, label="precies 1/4")
    ax.set_xticks(range(4), POSITIES); ax.set_ylim(0, 0.42)
    ax.set_title("vanaf '=', over 100 sommen", fontsize=10)
    ax.set_ylabel("aandachtsgewicht")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.97, 0.93, f"foutbalk x50\nwerkelijke spreiding\n{rij.std(0).mean():.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#4a5764")

    fig.suptitle("De aandacht is uniform: elke positie krijgt een kwart, bij elke som",
                 fontsize=12.5, y=1.04)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_value(bestand):
    """Waar het getal echt zit: de value-vectoren, met en zonder layernorm."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    waarde = torch.arange(10, dtype=torch.float)
    for ln, kleur, stijl in ((False, "#2a6f97", "o-"), (True, "#b0574f", "s--")):
        model, _, _, _ = leg_klaar(seed=0, gebruik_layernorm=ln)
        proj, var = value_van_cijfers(model)
        proj = proj / proj.abs().max()   # op gelijke schaal, alleen de VORM telt
        r = torch.corrcoef(torch.stack([proj, waarde]))[0, 1].abs()
        ax.plot(range(10), proj, stijl, color=kleur, lw=1.8, ms=7,
                label=f"layernorm {'aan' if ln else 'uit'}   (r={r:.4f}, as 1 = {var:.0%})")
    ax.set_xlabel("het cijfer"); ax.set_ylabel("value-vector op hoofdas (geschaald)")
    ax.set_title("Het getal zit in de value-vector, niet in de aandacht", fontsize=12)
    ax.set_xticks(range(10)); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def blokkeertest(model, x, tw):
    """Vervang de aandacht door precies uniform en kijk of er iets verandert.

    Als het mechanisme klopt - aandacht is een vaste middelingsstap, geen
    opzoeking - dan mag dit niets uitmaken.
    """
    som = tw[x[:, 0]] + tw[x[:, 2]]
    laag = model.lagen[0].attentie
    echt = model(x)[0].squeeze(-1)

    origineel = laag.forward
    def uniform(h):
        B, T, C = h.shape
        v = laag._splits(laag.V(h))
        g = torch.zeros(B, laag.n_koppen, T, T)
        for i in range(T):
            g[:, :, i, :i + 1] = 1.0 / (i + 1)
        u = (g @ v).transpose(1, 2).contiguous().view(B, T, C)
        return laag.uit_proj(u), g[:, 0]
    laag.forward = uniform
    plat = model(x)[0].squeeze(-1)
    laag.forward = origineel
    return echt, plat, som


if __name__ == "__main__":
    FIGUREN.mkdir(exist_ok=True)
    model, tok, x, tw = leg_klaar(seed=0)
    som = tw[x[:, 0]] + tw[x[:, 2]]

    gew, aff = aandacht(model, x)
    rij = gew[:, :, 3, :]
    print("aandacht vanaf de '='-positie, gemiddeld over 100 sommen:")
    print(f"  {'':>8}" + "".join(f"{p:>9}" for p in POSITIES))
    for kop in range(4):
        print(f"  kop {kop}:  " + "".join(f"{v:>9.3f}" for v in rij[:, kop].mean(0)))
    print(f"  spreiding over de sommen: {rij.std(0).mean():.5f}  (0 = volstrekt vast)")
    print(f"  affiniteiten voor de softmax liggen tussen "
          f"{aff[aff > -1e9].min():+.4f} en {aff[aff > -1e9].max():+.4f}")

    laag = model.lagen[0].attentie
    h = invoer_van_de_laag(model, x)
    with torch.no_grad():
        print(f"\ngemiddelde norm per positie:")
        print(f"  {'positie':>9}{'|q|':>9}{'|k|':>9}{'|v|':>9}")
        for p, naam in enumerate(POSITIES):
            print(f"  {naam:>9}{laag.Q(h)[:, p].norm(dim=1).mean():>9.4f}"
                  f"{laag.K(h)[:, p].norm(dim=1).mean():>9.4f}"
                  f"{laag.V(h)[:, p].norm(dim=1).mean():>9.4f}")

    proj, var = value_van_cijfers(model)
    waarde = torch.arange(10, dtype=torch.float)
    stappen = proj[1:] - proj[:-1]
    print(f"\nde value-vectoren van 0..9:")
    print(f"  variantie op as 1             : {var:.2%}")
    print(f"  correlatie met de getalwaarde : "
          f"{torch.corrcoef(torch.stack([proj, waarde]))[0, 1]:+.4f}")
    print(f"  stapgrootte                   : {stappen.mean():.4f} "
          f"+/- {stappen.std():.4f}")

    echt, plat, som = blokkeertest(model, x, tw)
    print(f"\nblokkeertest - aandacht hard op 1/4 gezet:")
    print(f"  grootste verandering in de uitvoer : {(plat - echt).abs().max():.5f}")
    print(f"  alle 100 sommen nog steeds goed    : {bool((plat.round() == som).all())}")

    figuur_aandacht(model, x, FIGUREN / "aandacht.png")
    print("\nvalue-vectoren met en zonder layernorm:")
    figuur_value(FIGUREN / "value.png")
    print(f"\nfiguren geschreven naar {FIGUREN}")
