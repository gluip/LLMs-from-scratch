"""Is n_embed=160 echt slechter, of alleen te hard getraind?

embed_vgl.py liet zien dat n_embed=160 bij LEERRATE=1e-2 overal slechter uitkomt
dan 80 - ook op de *train* loss (1,50 tegen 1,23), dus het is onderfitting en
geen overfitting. De verdenking is dat 1e-2 te grof is voor een model van 1,6M
parameters.

Dit script draait daarom lr x n_embed x venster. De 80-rij is de controle: zakt
die bij lagere lr net zo hard, dan lag het aan de leerrate en niet aan de
modelgrootte, en zegt dit niets over groter worden. Alleen als 160 bij lagere lr
onder 80 duikt, is een groter model daadwerkelijk de moeite waard.

De lr=1e-2 kolom komt uit embed_resultaten.pt, dus die hoeft niet opnieuw.
"""
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import (loss_per_positie, train_affiniteitsmodel, APPARAAT, DATA_MAP,
                 TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer)

N_STAPPEN = 5000
UITVOER = Path(__file__).parent / "lr_resultaten.pt"
EERDER = Path(__file__).parent / "embed_resultaten.pt"   # de lr=1e-2 metingen
EMBEDS = [80, 160]
LEERRATES = [3e-3, 5e-3]
VENSTERS = [64, 128]

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tok = CharTokenizer("".join(t for _, t in boeken))
tr, te = [], []
for _, t in boeken:
    d = torch.tensor(tok.encode(t), dtype=torch.long)
    s = int(TRAIN_FRACTIE * len(d))
    tr.append(d[:s]); te.append(d[s:])
train_ids, test_ids = torch.cat(tr).to(APPARAAT), torch.cat(te).to(APPARAAT)
print(f"apparaat={APPARAAT}"
      + (f" ({torch.cuda.get_device_name(0)})" if APPARAAT.type == "cuda" else "")
      + f"  vocab={tok.vocab_size} train={len(train_ids)}\n", flush=True)

# sleutel is (n_embed, lr, venster)
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
if EERDER.exists():  # de al gedraaide lr=1e-2 metingen erbij trekken
    for (n_embed, lengte), pp in torch.load(EERDER).items():
        resultaten.setdefault((n_embed, 1e-2, lengte), pp)

for n_embed in EMBEDS:
    for lr in LEERRATES:
        for lengte in VENSTERS:
            sleutel = (n_embed, lr, lengte)
            if sleutel in resultaten:
                continue
            t0 = time.time()
            model, train_losses, *_ = train_affiniteitsmodel(
                n_lagen=5, n_embed=n_embed, lengte=lengte, dropout=0.0,
                gebruik_layernorm=True, lr=lr, n_stappen=N_STAPPEN,
                train_ids=train_ids, test_ids=test_ids, tokenizer=tok,
                eval_interval=10**9)
            pp = loss_per_positie(model, test_ids, tok, lengte)
            resultaten[sleutel] = pp
            # de train loss erbij: onderfitting zie je daaraan, niet aan de test loss
            eind_train = sum(train_losses[-100:]) / 100
            print(f"n_embed {n_embed:>3d} lr {lr:.0e} venster {lengte:>4d}  "
                  f"laatste positie {pp[-1]:.4f}  train {eind_train:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(resultaten, UITVOER)


# ---------------------------------------------------------------------------
# Plot: per venster een paneel, x = leerrate, een lijn per modelgrootte
# ---------------------------------------------------------------------------
alle_lrs = sorted(LEERRATES + [1e-2])
fig, assen = plt.subplots(1, len(VENSTERS), figsize=(6.5 * len(VENSTERS), 5), sharey=True)
for ax, lengte in zip(assen, VENSTERS):
    for n_embed, kleur, stijl in zip(EMBEDS, ["tab:blue", "tab:orange"], ["-", "--"]):
        lrs = [lr for lr in alle_lrs if (n_embed, lr, lengte) in resultaten]
        waarden = [resultaten[(n_embed, lr, lengte)][-1].item() for lr in lrs]
        ax.plot(lrs, waarden, marker="o", color=kleur, linestyle=stijl,
                label=f"n_embed {n_embed}")
        for lr, y in zip(lrs, waarden):
            ax.annotate(f"{y:.3f}", (lr, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=kleur)
    ax.set_xscale("log")
    ax.set_xticks(alle_lrs); ax.set_xticklabels([f"{lr:g}" for lr in alle_lrs])
    ax.set_xlabel("leerrate")
    ax.set_title(f"venster {lengte}")
    ax.grid(alpha=0.3)
assen[0].set_ylabel("loss op de laatste positie (nats)")
assen[0].legend()

fig.suptitle("is 160 echt slechter, of alleen te hard getraind?")
fig.tight_layout()
plot_pad = Path(__file__).parent / "lr_vgl.png"
fig.savefig(plot_pad, dpi=130)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
