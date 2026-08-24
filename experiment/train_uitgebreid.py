"""Traint de beste gevonden configuratie (zie dropout_vgl.py) opnieuw, nu op de
volledige, uitgebreide dataset in TEKST_BESTANDEN in plaats van drie boeken.

langer_trainen.py liet zien dat de drie oorspronkelijke boeken (1,8M karakters)
bij 18000 stappen al ~45x herhaald werden en dat de test-loss vanaf stap ~10000
plat lag - meer stappen hielp niet meer, de data was op. Er zijn negentien DBNL-
titels bij gezet (Multatuli, Couperus, Van Lennep, Van Eeden, en meer), samen
goed voor 18,7M karakters - 10,4x de oorspronkelijke dataset. Bij 18000 stappen
is dat nog maar ~4,4x herhaling in plaats van 45x, dus ruim binnen wat gezond is.

Model wordt opgeslagen als model.pt (de oude 3-boeken-versie staat in
model_3boeken.pt); loss-curve als beste_loss_5boeken.png.
"""
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import (train_affiniteitsmodel, loss_per_positie, genereer, APPARAAT,
                 DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer,
                 N_LAGEN, N_KOPPEN, GEBRUIK_POSITIE, GEBRUIK_FEEDFORWARD,
                 LOSSE_QK, LOSSE_V, GEBRUIK_LAYERNORM, GEBRUIK_MASKER, UIT_PROJECTIE)

N_EMBED, LR, DROPOUT, LENGTE, N_STAPPEN = 160, 5e-3, 0.1, 64, 18000
EVAL_INTERVAL = 200
MODEL_PAD = Path(__file__).parent / "model.pt"

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
for naam, t in boeken:
    print(f"  {naam:24s} {len(t):>9,d} karakters")
tekst = "".join(t for _, t in boeken)
tok = CharTokenizer(tekst)
print(f"totaal: {len(tekst):,} karakters, vocab {tok.vocab_size}\n", flush=True)

tr, te = [], []
for _, t in boeken:
    d = torch.tensor(tok.encode(t), dtype=torch.long)
    s = int(TRAIN_FRACTIE * len(d))
    tr.append(d[:s]); te.append(d[s:])
train_ids, test_ids = torch.cat(tr).to(APPARAAT), torch.cat(te).to(APPARAAT)
print(f"apparaat={APPARAAT}"
      + (f" ({torch.cuda.get_device_name(0)})" if APPARAAT.type == "cuda" else "")
      + f"  train={len(train_ids):,}  (was 1.623.458 op drie boeken)\n", flush=True)

model, train_losses, test_stappen, test_losses = train_affiniteitsmodel(
    n_lagen=N_LAGEN, n_embed=N_EMBED, lengte=LENGTE, dropout=DROPOUT,
    gebruik_layernorm=GEBRUIK_LAYERNORM, lr=LR, n_stappen=N_STAPPEN,
    train_ids=train_ids, test_ids=test_ids, tokenizer=tok, eval_interval=EVAL_INTERVAL)

pp = loss_per_positie(model, test_ids, tok, LENGTE)
print(f"\nloss op de laatste positie: {pp[-1]:.4f}  (was 1,2455 op drie boeken)", flush=True)

max_loss = torch.log(torch.tensor(float(tok.vocab_size)))
plt.figure(figsize=(9, 6))
plt.plot(range(len(train_losses)), train_losses, alpha=0.3, label="train")
plt.plot(test_stappen, test_losses, marker="o", markersize=3, label="test")
plt.axhline(max_loss.item(), color="gray", linestyle="--", label="willekeurig gokken")
plt.xlabel("stap"); plt.ylabel("loss")
plt.title(f"train/test loss, 14 boeken (modern Nederlands) (n_embed={N_EMBED}, lr={LR}, dropout={DROPOUT}, lengte={LENGTE})")
plt.legend(); plt.tight_layout()
plot_pad = Path(__file__).parent / "beste_loss_14boeken.png"
plt.savefig(plot_pad)
print(f"plot opgeslagen: {plot_pad}", flush=True)

print("\nvoorbeeld:")
print(genereer(model, tok, start="Op een dag ", n_nieuw=200, lengte=LENGTE, temperatuur=0.8))

torch.save({
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "chars": tok.chars,
    "config": dict(n_embed=N_EMBED, n_lagen=N_LAGEN, n_koppen=N_KOPPEN,
                   gebruik_positie=GEBRUIK_POSITIE, gebruik_feedforward=GEBRUIK_FEEDFORWARD,
                   losse_qk=LOSSE_QK, losse_v=LOSSE_V, gebruik_layernorm=GEBRUIK_LAYERNORM,
                   gebruik_masker=GEBRUIK_MASKER, uit_projectie=UIT_PROJECTIE, dropout=0.0,
                   max_lengte=LENGTE),
    "lengte": LENGTE,
}, MODEL_PAD)
print(f"model opgeslagen: {MODEL_PAD}", flush=True)
