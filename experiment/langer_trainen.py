"""Train/test loss over de training heen, voor de beste gevonden configuratie
(n_embed=160, lr=5e-3, dropout=0.1, lengte=64 - zie dropout_vgl.py).

De sweeps bewaarden alleen de eindwaarden, niet de curve per stap, dus deze
configuratie wordt hier nog eens getraind met een fijnmazige eval_interval.
Zelfde plotstijl als de train/test-plot onderaan exp.py.
"""
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import (train_affiniteitsmodel, APPARAAT, DATA_MAP, TEKST_BESTANDEN,
                 TRAIN_FRACTIE, CharTokenizer)

N_EMBED, LR, DROPOUT, LENGTE, N_STAPPEN = 160, 5e-3, 0.1, 64, 36000
EVAL_INTERVAL = 200  # fijner dan de 3000 uit dropout_vgl.py, voor een leesbare curve

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

model, train_losses, test_stappen, test_losses = train_affiniteitsmodel(
    n_lagen=5, n_embed=N_EMBED, lengte=LENGTE, dropout=DROPOUT, gebruik_layernorm=True,
    lr=LR, n_stappen=N_STAPPEN, train_ids=train_ids, test_ids=test_ids, tokenizer=tok,
    eval_interval=EVAL_INTERVAL)

max_loss = torch.log(torch.tensor(float(tok.vocab_size)))
plt.figure(figsize=(9, 6))
plt.plot(range(len(train_losses)), train_losses, alpha=0.3, label="train")
plt.plot(test_stappen, test_losses, marker="o", markersize=3, label="test")
plt.axhline(max_loss.item(), color="gray", linestyle="--", label="willekeurig gokken")
plt.xlabel("stap")
plt.ylabel("loss")
plt.title(f"train/test loss (n_embed={N_EMBED}, lr={LR}, dropout={DROPOUT}, lengte={LENGTE})")
plt.legend()
plt.tight_layout()
plot_pad = Path(__file__).parent / "beste_loss_36k.png"
plt.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
print(f"laatste test loss: {test_losses[-1]:.4f}", flush=True)
