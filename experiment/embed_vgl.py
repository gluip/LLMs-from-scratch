"""Ligt het plateau bij venster 64 aan het model of aan de taal?

venster_vgl.py liet zien dat bij n_embed=80 de winst na venster 64 op is: 64 ->
128 leverde op de laatste positie nog maar 0,005 nats. Twee verklaringen passen
daarbij, en die kun je niet uit elkaar houden met één modelgrootte:

  1. de taal: in Nederlands proza zit op karakterniveau vrijwel alle informatie
     in het huidige woord en het vorige, en verder terug staat simpelweg niets
     bruikbaars meer;
  2. het model: 80 dimensies zijn te weinig om verbanden over 100 karakters
     afstand vast te houden, ook al zijn ze er wel.

Dit script draait daarom een raster van n_embed x venster. Schuift het plateau
mee naar rechts als het model groter wordt, dan was het model de rem (2).
Blijven de curves op dezelfde plek afvlakken, dan is het de taal (1).
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
UITVOER = Path(__file__).parent / "embed_resultaten.pt"
EMBEDS = [80, 160]
VENSTERS = [32, 64, 128]
# n_koppen blijft 4, dus 20 resp. 40 dimensies per kop. Alles verder gelijk aan
# venster_vgl.py, zodat het enige verschil tussen de twee rijen n_embed is.

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

# sleutel is (n_embed, venster); al berekende combinaties worden overgeslagen
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
for n_embed in EMBEDS:
    for lengte in VENSTERS:
        sleutel = (n_embed, lengte)
        if sleutel in resultaten:
            pp = resultaten[sleutel]
            print(f"n_embed {n_embed:>3d} venster {lengte:>4d}  laatste positie {pp[-1]:.4f}  "
                  f"(overgeslagen)", flush=True)
            continue
        t0 = time.time()
        model, *_ = train_affiniteitsmodel(
            n_lagen=5, n_embed=n_embed, lengte=lengte, dropout=0.0, gebruik_layernorm=True,
            n_stappen=N_STAPPEN, train_ids=train_ids, test_ids=test_ids, tokenizer=tok,
            eval_interval=10**9)
        pp = loss_per_positie(model, test_ids, tok, lengte)
        resultaten[sleutel] = pp
        print(f"n_embed {n_embed:>3d} venster {lengte:>4d}  laatste positie {pp[-1]:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(resultaten, UITVOER)  # na elke combinatie, zodat een afgebroken run niet weg is


# ---------------------------------------------------------------------------
# Plot: per modelgrootte een lijn door de vensters heen
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
for n_embed, kleur, stijl in zip(EMBEDS, ["tab:blue", "tab:orange"], ["-", "--"]):
    vensters = [l for l in VENSTERS if (n_embed, l) in resultaten]
    if not vensters:
        continue
    # de laatste positie, niet het gemiddelde: die is de eerlijke vergelijking,
    # want het gemiddelde vleit grote vensters (minder dure beginposities)
    laatste = [resultaten[(n_embed, l)][-1].item() for l in vensters]
    ax.plot(vensters, laatste, marker="o", color=kleur, linestyle=stijl,
            label=f"n_embed {n_embed}")
    for l, y in zip(vensters, laatste):
        ax.annotate(f"{y:.3f}", (l, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=kleur)

ax.set_xscale("log", base=2)
ax.set_xticks(VENSTERS); ax.set_xticklabels(VENSTERS)
ax.set_xlabel("venstergrootte")
ax.set_ylabel("loss op de laatste positie (nats)")
ax.set_title("schuift het plateau op met een groter model?")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
plot_pad = Path(__file__).parent / "embed_vgl.png"
fig.savefig(plot_pad, dpi=130)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
