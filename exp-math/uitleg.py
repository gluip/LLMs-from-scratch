# Hoe rekent dit netwerk? Alle stappen van één som, met echte getallen.
#
# Dit script is de onderbouwing van verslag-machinerie.html. Het rekent de
# hele keten na voor het kleinste optel-model (49 parameters) en tekent de
# twee figuren die de uitleg dragen.
#
# Draaien:  .venv/bin/python -u exp-math/uitleg.py

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

# de kleinste configuratie per bewerking waarmee alle achtergehouden sommen
# goed gaan; zie EXPERIMENTEN.md voor hoe die gevonden zijn
KAAL = dict(n_embed=2)                                             # optellen, 49 parameters
VOL = dict(n_embed=16, positie=True, leer_aandacht=True, uit_proj=True)
BEIDE = dict(n_embed=16, positie=True, leer_aandacht=True, uit_proj=True, n_koppen=2)


def getraind(bewerking, seed=0, **knoppen):
    tok, tw, xtr, ytr, xte, yte = r.laad(bewerking)
    model, _, acc = r.train(xtr, ytr, xte, yte, tok.vocab_size, seed=seed, **knoppen)
    return model, tok, acc


@torch.no_grad()
def schaal(model, tok):
    """s(token) = uit(V(embedding)): het getal dat één token in de uitvoer legt.

    De hele keten van een token naar de uitvoer is lineair, dus valt hij samen
    te vatten in één getal per token. Voor de cijfers loopt dat getal recht
    evenredig op met de cijferwaarde - dat is waarom middelen optellen wordt.
    """
    return model.V(model.embed.weight) @ model.uit.weight[0]


@torch.no_grad()
def aandacht(model, tok, teken, sommen, kop=0):
    """Aandachtsverdeling vanaf de '='-positie, voor een lijst (a, b)-paren."""
    x = torch.tensor([tok.encode([str(a), teken, str(b), "="]) for a, b in sommen])
    h = model.embed(x)
    if model.pos is not None:
        h = h + model.pos(torch.arange(4))
    q, k = model._splits(model.Q(h)), model._splits(model.K(h))
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(model.kop_dim), -1)[:, kop, 3, :]


def figuur_schaal(bestand):
    """s(cijfer) tegen de cijferwaarde: een rechte lijn met gelijke stappen."""
    model, tok, _ = getraind("optellen", **KAAL)
    s = schaal(model, tok)
    cijfers = torch.arange(10, dtype=torch.float)
    A = torch.stack([cijfers, torch.ones(10)], 1)
    hel, snij = torch.linalg.lstsq(A, s[:10].unsqueeze(1)).solution.squeeze(1)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(cijfers, hel * cijfers + snij, "-", color="#b8c4cc", lw=6, alpha=0.6,
            label=f"rechte lijn: {hel:+.2f}·d {snij:+.1f}")
    ax.plot(cijfers, s[:10], "o", color="#2a6f97", ms=9, label="s(cijfer), gemeten")
    for t, kleur in (("+", "#c08a2e"), ("=", "#b0574f")):
        ax.axhline(s[tok.naar_int[t]], color=kleur, ls=":", lw=1.6)
        ax.annotate(f"s('{t}') = {s[tok.naar_int[t]]:.1f}  (vast)", (9.3, s[tok.naar_int[t]]),
                    color=kleur, fontsize=9, va="center")
    ax.set_xlabel("het cijfer"); ax.set_ylabel("s(token) — bijdrage aan de uitvoer")
    ax.set_xticks(range(10)); ax.set_xlim(-0.6, 12.5); ax.grid(alpha=0.3)
    ax.set_title("Elk cijfer legt een bedrag in dat recht evenredig is met zijn waarde",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)
    return hel, snij, s


def figuur_drie_regimes(bestand):
    """Middelen, kiezen, schakelen: wat de aandacht in de drie taken doet."""
    sommen = [(9, 0), (0, 9), (4, 5)]
    kolommen = [
        ("optellen\nde aandacht MIDDELT", "optellen", "+", VOL, 0),
        ("aftrekken\nde aandacht KIEST", "aftrekken", "-", VOL, 0),
        ("allebei\nde aandacht SCHAKELT", "beide", None, BEIDE, 1),
    ]
    labels = ["a", "teken", "b", "="]
    kleuren = ["#2a6f97", "#c08a2e", "#b0574f", "#b8c4cc"]

    fig, assen = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, (titel, bewerking, teken, knoppen, kop) in zip(assen, kolommen):
        model, tok, acc = getraind(bewerking, **knoppen)
        if teken is None:
            # bij 'beide' vergelijken we de twee bewerkingen, en dan is het
            # GEMIDDELDE over alle honderd sommen de eerlijke maat: bij losse
            # sommen varieert de omschakeling sterk, en dan kies je er zomaar
            # een uit die het verhaal vleit
            alle = [(a, b) for a in range(10) for b in range(10)]
            g = torch.stack([aandacht(model, tok, t, alle, kop=kop).mean(0)
                             for t in ("+", "-")])
            namen = ["optellen\n(gemiddeld)", "aftrekken\n(gemiddeld)"]
        else:
            g = aandacht(model, tok, teken, sommen, kop=kop)
            namen = [f"{a} {teken} {b}" for a, b in sommen]
        onder = torch.zeros(len(namen))
        for j, (lbl, kleur) in enumerate(zip(labels, kleuren)):
            h = g[:, j]
            ax.bar(namen, h, 0.6, bottom=onder, color=kleur)
            for i, (hh, oo) in enumerate(zip(h, onder)):
                if hh > 0.1:
                    ax.text(i, oo + hh / 2, f"{lbl} {hh:.2f}", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            onder = onder + h
        ax.set_title(titel, fontsize=11)
        ax.set_ylim(0, 1)
    assen[0].set_ylabel("aandacht vanaf de '='-positie")
    fig.suptitle("Dezelfde laag, drie taken, drie soorten gedrag", fontsize=12.5, y=1.04)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def narekenen():
    """De hele keten voor '9 + 9 =', stap voor stap, met echte getallen."""
    model, tok, acc = getraind("optellen", **KAAL)
    n_par = sum(p.numel() for p in model.parameters())
    s = schaal(model, tok)
    w, bias = model.uit.weight[0], model.uit.bias[0]

    print(f"het kleinste optel-model: {n_par} parameters, {acc:.0%} op de "
          f"achtergehouden sommen\n")
    with torch.no_grad():
        x = torch.tensor([tok.encode(["9", "+", "9", "="])])
        h = model.embed(x)[0]
        v = model.V(h)
        gemiddeld = v.mean(0)
        laatst = h[-1] + gemiddeld

        print("'9 + 9 =' stap voor stap:")
        print(f"  1  tokens          ['9', '+', '9', '=']")
        print(f"  2  embeddings      " + "  ".join(f"[{e[0]:+.2f},{e[1]:+.2f}]" for e in h))
        print(f"  3  values V(h)     " + "  ".join(f"[{e[0]:+.2f},{e[1]:+.2f}]" for e in v))
        print(f"  4  elk x 0,25      gemiddelde = [{gemiddeld[0]:+.3f}, {gemiddeld[1]:+.3f}]")
        print(f"  5  + h['=']        [{laatst[0]:+.3f}, {laatst[1]:+.3f}]")
        print(f"  6  uitlezen        {model.uit(laatst).item():.3f}  ->  "
              f"afgerond {model.uit(laatst).round().item():.0f}")

        print(f"\nhetzelfde in bedragen per token (s = uit(V(embedding))):")
        for t in ("9", "+", "9", "="):
            print(f"  s('{t}') = {s[tok.naar_int[t]]:>8.3f}")
        mid = sum(s[tok.naar_int[t]] for t in ("9", "+", "9", "=")) / 4
        vast = (w @ h[-1]) + bias
        print(f"  {'gemiddeld':>9} = {mid:>8.3f}")
        print(f"  {'+ vast':>9} = {vast:>8.3f}   (uit(h['=']) plus de bias)")
        print(f"  {'':>9}   {'':>8}   {'-'*8}")
        print(f"  {'uitvoer':>9} = {mid + vast:>8.3f}")

        cijfers = torch.arange(10, dtype=torch.float)
        A = torch.stack([cijfers, torch.ones(10)], 1)
        hel, snij = torch.linalg.lstsq(A, s[:10].unsqueeze(1)).solution.squeeze(1)
        e_is = model.embed.weight[tok.naar_int["="]]
        c = snij / 2 + (s[tok.naar_int["+"]] + s[tok.naar_int["="]]) / 4 + (w @ e_is) + bias
        print(f"\nen dus is het hele model, alle 49 parameters samen:")
        print(f"  uitvoer = {hel/4:+.4f}·(a+b) {c:+.4f}")

        tok2, tw, xtr, ytr, xte, yte = r.laad("optellen")
        som = tw[xtr[:, 0]] + tw[xtr[:, 2]]
        echt = model(xtr)
        print(f"  grootste afwijking van die formule, over alle 80 sommen: "
              f"{(echt - ((hel/4)*som + c)).abs().max():.4f}")
        print(f"  grootste afwijking van (a+b) zelf:                       "
              f"{(echt - som).abs().max():.4f}")


if __name__ == "__main__":
    FIGUREN.mkdir(exist_ok=True)
    narekenen()
    print("\nfiguren maken...")
    figuur_schaal(FIGUREN / "uitleg_schaal.png")
    figuur_drie_regimes(FIGUREN / "uitleg_regimes.png")
    print(f"geschreven naar {FIGUREN}")
