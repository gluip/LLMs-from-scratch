# Zit er semantiek in de gewichten? Oftewel: heeft het model een getallenlijn?
#
# Het model krijgt de cijfers 0..9 als losse, betekenisloze symbolen binnen.
# Niets vertelt het dat 7 groter is dan 3, of dat 3 tussen 2 en 4 in ligt. Als
# die ordening tóch in de geleerde embeddings terug te vinden is, heeft het
# model iets over getallen begrepen in plaats van 80 regels onthouden.
#
# Draaien:  .venv/bin/python -u exp-math/embeddings_kijken.py
# Schrijft de figuren naar exp-math/figuren/.

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
import wiskunde as w

FIGUREN = Path(__file__).parent / "figuren"
SEEDS = range(6)
KLEUR = plt.get_cmap("viridis")


def hoofdvlak(E):
    """Projecteer de embeddings op hun eigen twee belangrijkste assen (PCA).

    Geeft de 2D-coordinaten terug plus de verklaarde variantie per as. Let op
    dat je hiervoor de volledige SVD nodig hebt: torch.pca_lowrank(q=2) geeft
    per definitie 100% terug omdat het maar twee componenten kent, en dan meet
    je je eigen keuze in plaats van de data.
    """
    E = E - E.mean(0)
    _, S, Vh = torch.linalg.svd(E, full_matrices=False)
    return E @ Vh.T[:, :2], S ** 2 / (S ** 2).sum()


def hoeken(punten):
    """Hoek van elk punt in het vlak, oplopend vanaf cijfer 0.

    De modulo laat de reeks doorlopen in plaats van bij 180 graden om te
    klappen; zonder dat lijkt een keurige boog een sprong te maken.
    """
    h = torch.atan2(punten[:, 1], punten[:, 0]) * 180 / math.pi
    h = (h - h[0]) % 360
    return torch.where(h.diff(prepend=h[:1]) < -180, h + 360, h)


def train_alles(seeds=SEEDS, **knoppen):
    regels = w.laad_sommen()
    tok = w.SomTokenizer(regels)
    ids = torch.tensor([tok.encode(r) for r in regels], dtype=torch.long)
    x, y = ids[:, :w.LENGTE], ids[:, w.LENGTE]
    xtr, ytr, xte, yte = w.splits_train_test(x, y)
    tw = tok.waarden()
    uit = []
    for seed in seeds:
        model, g = w.train(xtr, ytr, tw[ytr], xte, yte, tw[yte], tok, tw,
                           seed=seed, stil=True, **knoppen)
        uit.append((model, g[-1][2]))
        print(f"  seed {seed}: test {g[-1][2]:.0%}")
    return tok, uit


def figuur_boog(modellen, bestand):
    """De cijfers 0..9 in hun eigen hoofdvlak, verbonden op volgorde."""
    fig, assen = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (seed, (model, _)) in zip(assen, enumerate(modellen[:3])):
        p, var = hoofdvlak(model.embed.weight.detach()[:10])
        ax.plot(p[:, 0], p[:, 1], "-", color="0.75", lw=1.4, zorder=1)
        ax.scatter(p[:, 0], p[:, 1], c=range(10), cmap=KLEUR, s=260, zorder=2)
        for d in range(10):
            ax.annotate(str(d), (p[d, 0], p[d, 1]), ha="center", va="center",
                        color="white", fontweight="bold", fontsize=11, zorder=3)
        ax.set_title(f"seed {seed}   (PC1 {var[0]:.0%} + PC2 {var[1]:.0%})", fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    fig.suptitle("De geleerde embeddings van 0 t/m 9, geprojecteerd op hun twee hoofdassen",
                 fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_hoek(modellen, bestand):
    """Hoek langs de boog tegen de getalwaarde: recht = gelijkmatige stappen."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    waarde = torch.arange(10, dtype=torch.float)
    for seed, (model, _) in enumerate(modellen):
        p, _ = hoofdvlak(model.embed.weight.detach()[:10])
        h = hoeken(p)
        r = torch.corrcoef(torch.stack([h, waarde]))[0, 1]
        ax.plot(range(10), h, "o-", alpha=0.8, lw=1.6, ms=6, label=f"seed {seed}  (r={r:+.3f})")
    ax.set_xlabel("het cijfer"); ax.set_ylabel("hoek langs de boog (graden)")
    ax.set_title("De hoek loopt recht evenredig op met de getalwaarde", fontsize=12)
    ax.set_xticks(range(10)); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_variantie(modellen, bestand):
    """Hoeveel van de 32 dimensies gebruikt het model echt?"""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    stapels = torch.stack([hoofdvlak(m.embed.weight.detach()[:10])[1] for m, _ in modellen])
    gem, spreiding = stapels.mean(0), stapels.std(0)
    n = min(10, len(gem))
    ax.bar(range(1, n + 1), gem[:n] * 100, yerr=spreiding[:n] * 100,
           color=["#2a6f97"] * 2 + ["#b8c4cc"] * (n - 2), capsize=3)
    ax.set_xlabel("hoofdas (PC)"); ax.set_ylabel("verklaarde variantie (%)")
    ax.set_title(f"Twee assen dragen {(gem[0] + gem[1]) * 100:.0f}% — de rest is ruis",
                 fontsize=12)
    ax.set_xticks(range(1, n + 1)); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_afstand(modellen, bestand):
    """Afstand tussen embeddings tegen het verschil in waarde."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    per_verschil = {v: [] for v in range(1, 10)}
    for model, _ in modellen:
        E = model.embed.weight.detach()[:10]
        E = E - E.mean(0)
        for a in range(10):
            for b in range(a + 1, 10):
                per_verschil[b - a].append((E[a] - E[b]).norm().item())
    gem = [torch.tensor(per_verschil[v]).mean() for v in range(1, 10)]
    sd = [torch.tensor(per_verschil[v]).std() for v in range(1, 10)]
    ax.errorbar(range(1, 10), gem, yerr=sd, fmt="o-", color="#2a6f97", capsize=3, lw=1.8)
    ax.set_xlabel("verschil in waarde |a - b|")
    ax.set_ylabel("afstand tussen de embeddings")
    ax.set_title("Verder uit elkaar in waarde = verder uit elkaar in de ruimte\n"
                 "(de afvlakking hoort bij een boog, niet bij een rechte lijn)", fontsize=11)
    ax.set_xticks(range(1, 10)); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_stabiliteit(bestand):
    """Test-accuratesse per seed, met en zonder stevige weight decay."""
    print("\n  wd=0.01 (AdamW default), 3000 stappen:")
    _, zonder = train_alles(range(10), weight_decay=0.01, n_stappen=3000)
    print("  wd=0.3, 10000 stappen:")
    _, met = train_alles(range(10), weight_decay=0.3, n_stappen=10000)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = torch.arange(10)
    ax.bar(x - 0.2, [a * 100 for _, a in zonder], 0.4, label="wd=0,01 (default)", color="#c96f6f")
    ax.bar(x + 0.2, [a * 100 for _, a in met], 0.4, label="wd=0,3", color="#2a6f97")
    ax.set_xlabel("seed"); ax.set_ylabel("goed van de 20 achtergehouden sommen (%)")
    ax.set_title("Weight decay haalt de ondergrens omhoog", fontsize=12)
    ax.set_xticks(x.tolist()); ax.set_ylim(0, 118); ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.95)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


def figuur_layernorm(bestand):
    """Boog versus rechte lijn: wat layernorm met de geometrie doet.

    Layernorm normaliseert de lengte van elke vector weg. Wat overblijft om
    informatie in te stoppen is de richting, en richtingen liggen op een
    cirkel - vandaar de boog. Zonder layernorm mag de lengte meedoen en wordt
    het een rechte, gelijkmatig verdeelde lijn.
    """
    fig, assen = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, ln in zip(assen, (True, False)):
        print(f"  layernorm={ln}:")
        _, modellen = train_alles(range(1), gebruik_layernorm=ln)
        p, var = hoofdvlak(modellen[0][0].embed.weight.detach()[:10])
        ax.plot(p[:, 0], p[:, 1], "-", color="0.75", lw=1.4, zorder=1)
        ax.scatter(p[:, 0], p[:, 1], c=range(10), cmap=KLEUR, s=250, zorder=2)
        for d in range(10):
            ax.annotate(str(d), (p[d, 0], p[d, 1]), ha="center", va="center",
                        color="white", fontweight="bold", fontsize=10, zorder=3)
        ax.set_title(f"{'met' if ln else 'zonder'} layernorm\n"
                     f"PC1 {var[0]:.0%} + PC2 {var[1]:.0%}", fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        # beide panelen vierkant en op hun eigen schaal: alleen zo is de VORM
        # te vergelijken. Zonder dit is het linkerplaatje 6x kleiner dan het
        # rechter en lijkt de rechte lijn een klein streepje.
        breedte = max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()) * 0.62
        mx, my = p[:, 0].mean(), p[:, 1].mean()
        ax.set_xlim(mx - breedte, mx + breedte); ax.set_ylim(my - breedte, my + breedte)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    fig.suptitle("Layernorm normaliseert de lengte weg, dus blijft alleen de richting over",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(bestand, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    FIGUREN.mkdir(exist_ok=True)
    print(f"trainen, {len(list(SEEDS))} seeds:")
    tok, modellen = train_alles()

    figuur_boog(modellen, FIGUREN / "boog.png")
    figuur_hoek(modellen, FIGUREN / "hoek.png")
    figuur_variantie(modellen, FIGUREN / "variantie.png")
    figuur_afstand(modellen, FIGUREN / "afstand.png")
    figuur_stabiliteit(FIGUREN / "stabiliteit.png")
    print("\nlayernorm aan/uit:")
    figuur_layernorm(FIGUREN / "layernorm.png")

    # De antwoord-tokens 10..18 zijn nooit invoer (het antwoord is een getal,
    # geen token, bij deze loss), dus ze krijgen nooit gradient. Weight decay
    # drukt ze daardoor plat - een mooie controle dat wd echt aanstaat.
    E = modellen[0][0].embed.weight.detach()
    invoer = [i for i, t in enumerate(tok.tokens) if t.isdigit() and int(t) <= 9]
    ongebruikt = [i for i, t in enumerate(tok.tokens) if t.isdigit() and int(t) > 9]
    print(f"\ngemiddelde norm van de embeddings:")
    print(f"  cijfers 0..9 (wel invoer) : {E[invoer].norm(dim=1).mean():.3f}")
    print(f"  10..18 (nooit invoer)     : {E[ongebruikt].norm(dim=1).mean():.3f}")

    print(f"\nfiguren geschreven naar {FIGUREN}")
