"""Absolute positie-embeddings tegen RoPE (rotary position embeddings).

Het huidige model.pt gebruikt gebruik_positie=True: elke absolute positie
0..63 krijgt een eigen geleerde vector, opgeteld bij de karakter-embedding.
Nadeel: positie 40 en 41 hebben totaal ongerelateerde vectoren, dus het model
moet zelf leren dat "40 is één minder dan 41" betekent. En voorbij max_lengte
bestaat er geen rij in de tabel - vandaar dat genereer() de context noodgedwongen
afkapt (zie de docstring daar).

RoPE (zie rope_hoeken/pas_rope_toe in exp.py) doet iets anders: geen tabel, geen
optelling bij de embedding. In plaats daarvan roteert het Q en K vlak voor het
inproduct, met een hoek die van hun positie afhangt. Het inproduct van twee
geroteerde vectoren hangt daarna alleen af van hun onderlinge afstand, niet van
hun absolute positie - een patroon dat drie karakters terug staat, hoeft dus
maar één keer geleerd te worden in plaats van apart per absolute positie.

Twee varianten, verder identiek aan de nu beste configuratie (n_embed=160,
lr=5e-3, dropout=0.1, lengte=64, 18000 stappen, losse Q/K/V) op de schone
14-boeken-dataset:
  - "zonder RoPE" = de huidige standaard (gebruik_positie=True, gebruik_rope=False),
    dient als sanity-check: moet dicht bij de bekende 1,3217 / 1,2781 uitkomen.
  - "met RoPE" = gebruik_positie=False, gebruik_rope=True. Niet allebei tegelijk
    aan (zie de docstring van AffiniteitsLaag): anders meet je niet meer wat
    RoPE alleen bijdraagt.
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
    ("zonder RoPE (huidig)", dict(gebruik_positie=True,  gebruik_rope=False)),
    ("met RoPE",             dict(gebruik_positie=False, gebruik_rope=True)),
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

UITVOER = Path(__file__).parent / "rope_resultaten.pt"
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
for naam, instellingen in VARIANTEN:
    if naam in resultaten:
        r = resultaten[naam]
        print(f"{naam:>22s}  test {r['test_stappen_losses'][-1]:.4f}  laatste positie {r['pp'][-1]:.4f}  "
              f"(overgeslagen)", flush=True)
        continue
    model, train_losses, test_stappen, test_losses = train_affiniteitsmodel(
        n_lagen=5, n_embed=N_EMBED, lengte=LENGTE, dropout=DROPOUT,
        gebruik_layernorm=True, lr=LR, n_stappen=N_STAPPEN, n_koppen=N_KOPPEN,
        losse_qk=True, losse_v=True, uit_projectie=True,
        train_ids=train_ids, test_ids=test_ids, tokenizer=tok, eval_interval=EVAL_INTERVAL,
        **instellingen)
    pp = loss_per_positie(model, test_ids, tok, LENGTE)
    n_par = sum(p.numel() for p in model.parameters())
    resultaten[naam] = dict(train_losses=train_losses, test_stappen=test_stappen,
                            test_stappen_losses=test_losses, pp=pp, n_par=n_par)
    print(f"\n{naam:>22s}  test {test_losses[-1]:.4f}  laatste positie {pp[-1]:.4f}  "
          f"({n_par:,} parameters)", flush=True)
    print("voorbeeld:", genereer(model, tok, start="Op een dag ", n_nieuw=150, lengte=LENGTE,
                                  temperatuur=0.8), flush=True)
    torch.save(resultaten, UITVOER)

print("\n" + "=" * 70)
for naam, _ in VARIANTEN:
    r = resultaten[naam]
    print(f"{naam:>22s}  test {r['test_stappen_losses'][-1]:.4f}  "
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
ax.set_title(f"absolute positie vs RoPE (n_embed={N_EMBED}, lr={LR}, dropout={DROPOUT}, lengte={LENGTE})")
ax.legend()
fig.tight_layout()
plot_pad = Path(__file__).parent / "rope_vgl.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
