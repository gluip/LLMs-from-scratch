"""Eén gedeelde matrix voor affiniteit+value tegen losse Q/K/V.

Met losse_qk=False, losse_v=False (de "alles-in-1"-versie waarmee AffiniteitsLaag
begon) wordt één matrix W gebruikt voor zowel "wie zoek ik" als "wie bied ik aan"
als "wat neem ik mee": elke vector wordt ermee geprojecteerd, en de affiniteit
tussen positie i en j is het inproduct van die ene projectie met zichzelf.
Nadeel: het inproduct van een vector met zichzelf is vrijwel altijd het grootst,
dus elke positie let vooral op zichzelf.

Met losse_qk=True + losse_v=True (het huidige model.pt) krijgen Q, K en V
allebei hun eigen matrix, zodat "waar ik naar zoek", "wat ik aanbied" en "wat ik
meeneem" losgekoppeld zijn - het volledige Q/K/V zoals bij echte attention.

Beide varianten draaien op de nu beste hyperparameters (n_embed=160, lr=5e-3,
dropout=0.1, lengte=64, 18000 stappen) op de schone 14-boeken-dataset.
"""
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import (train_affiniteitsmodel, loss_per_positie, genereer, APPARAAT,
                 DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer, N_KOPPEN)

N_EMBED, LR, DROPOUT, LENGTE, N_STAPPEN = 160, 5e-3, 0.1, 64, 18000
EVAL_INTERVAL = 200

VARIANTEN = [
    ("1 matrix (W)",  dict(losse_qk=False, losse_v=False)),
    ("losse Q/K/V",   dict(losse_qk=True,  losse_v=True)),
]

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
      + f"  vocab={tok.vocab_size} train={len(train_ids):,}\n", flush=True)

UITVOER = Path(__file__).parent / "qkv_resultaten.pt"
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
for naam, instellingen in VARIANTEN:
    if naam in resultaten:
        r = resultaten[naam]
        print(f"{naam:>14s}  test {r['test_stappen_losses'][-1]:.4f}  laatste positie {r['pp'][-1]:.4f}  "
              f"(overgeslagen)", flush=True)
        continue
    model, train_losses, test_stappen, test_losses = train_affiniteitsmodel(
        n_lagen=5, n_embed=N_EMBED, lengte=LENGTE, dropout=DROPOUT,
        gebruik_layernorm=True, lr=LR, n_stappen=N_STAPPEN, n_koppen=N_KOPPEN,
        train_ids=train_ids, test_ids=test_ids, tokenizer=tok, eval_interval=EVAL_INTERVAL,
        **instellingen)
    pp = loss_per_positie(model, test_ids, tok, LENGTE)
    n_par = sum(p.numel() for p in model.parameters())
    resultaten[naam] = dict(train_losses=train_losses, test_stappen=test_stappen,
                            test_stappen_losses=test_losses, pp=pp, n_par=n_par)
    print(f"\n{naam:>14s}  test {test_losses[-1]:.4f}  laatste positie {pp[-1]:.4f}  "
          f"({n_par:,} parameters)", flush=True)
    print("voorbeeld:", genereer(model, tok, start="Op een dag ", n_nieuw=150, lengte=LENGTE,
                                  temperatuur=0.8), flush=True)
    torch.save(resultaten, UITVOER)

print("\n" + "=" * 70)
for naam, _ in VARIANTEN:
    r = resultaten[naam]
    print(f"{naam:>14s}  test {r['test_stappen_losses'][-1]:.4f}  "
          f"laatste positie {r['pp'][-1]:.4f}  {r['n_par']:,} parameters")

# ---------------------------------------------------------------------------
# Plot: train/test loss over de training heen, één lijn per variant
# ---------------------------------------------------------------------------
max_loss = torch.log(torch.tensor(float(tok.vocab_size)))
fig, ax = plt.subplots(figsize=(9, 6))
for (naam, _), kleur in zip(VARIANTEN, ["tab:blue", "tab:orange"]):
    r = resultaten[naam]
    ax.plot(range(len(r["train_losses"])), r["train_losses"], color=kleur, alpha=0.25, linewidth=0.8)
    ax.plot(r["test_stappen"], r["test_stappen_losses"], color=kleur, marker="o", markersize=3,
            label=naam)
ax.axhline(max_loss.item(), color="gray", linestyle="--", label="willekeurig gokken")
ax.set_xlabel("stap"); ax.set_ylabel("loss")
ax.set_title(f"1 matrix vs losse Q/K/V (n_embed={N_EMBED}, lr={LR}, dropout={DROPOUT}, lengte={LENGTE})")
ax.legend()
fig.tight_layout()
plot_pad = Path(__file__).parent / "qkv_vgl.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
