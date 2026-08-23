"""De winnaar van de sweep tegen de huidige standaard, op de volle trainingsduur.

lr_vgl.py vond n_embed=160 bij lr=5e-3 als beste (1,315 tegen 1,375), maar dat
was op 5000 stappen. exp.py draait er 18000, en het optimum van de leerrate
schuift mee met de trainingsduur: langer trainen verdraagt doorgaans een lagere
lr, en met de cosine-decay erbij is 5e-3 over 18000 stappen een ander regime dan
over 5000. Daarom hier allebei opnieuw, op 18000 stappen.

Het beste model wordt weggeschreven als model.pt, zodat praat.py het meteen
gebruikt.
"""
import time
from pathlib import Path

import torch

from exp import (loss_per_positie, train_affiniteitsmodel, genereer, APPARAAT,
                 DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer,
                 N_LAGEN, N_KOPPEN, GEBRUIK_POSITIE, GEBRUIK_FEEDFORWARD,
                 LOSSE_QK, LOSSE_V, GEBRUIK_LAYERNORM, GEBRUIK_MASKER, UIT_PROJECTIE)

N_STAPPEN = 18000
LENGTE = 64          # in zes vergelijkingen won 64 het steeds van 128, zie lr_vgl.py
MODEL_PAD = Path(__file__).parent / "model.pt"

VARIANTEN = [
    ("standaard", dict(n_embed=80, lr=1e-2)),
    ("sweep-winnaar", dict(n_embed=160, lr=5e-3)),
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
      + f"  vocab={tok.vocab_size} train={len(train_ids)}\n", flush=True)

uitkomsten = {}
for naam, instellingen in VARIANTEN:
    t0 = time.time()
    model, train_losses, *_ = train_affiniteitsmodel(
        n_lagen=N_LAGEN, lengte=LENGTE, dropout=0.0, gebruik_layernorm=True,
        n_stappen=N_STAPPEN, train_ids=train_ids, test_ids=test_ids, tokenizer=tok,
        eval_interval=3000, **instellingen)
    pp = loss_per_positie(model, test_ids, tok, LENGTE)
    eind_train = sum(train_losses[-100:]) / 100
    uitkomsten[naam] = (pp[-1].item(), eind_train, model, instellingen)
    print(f"\n{naam:>14s}  n_embed {instellingen['n_embed']:>3d}  lr {instellingen['lr']:.0e}  "
          f"test {pp[-1]:.4f}  train {eind_train:.4f}  ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "=" * 70)
for naam, (test, train, _, inst) in uitkomsten.items():
    # het gat tussen train en test is de overfit-maat: klein gat = nog ruimte over
    print(f"{naam:>14s}  test {test:.4f}  train {train:.4f}  gat {test - train:+.4f}")

beste = min(uitkomsten, key=lambda naam: uitkomsten[naam][0])
beste_test, _, beste_model, beste_inst = uitkomsten[beste]
print(f"\nbeste: {beste} (test {beste_test:.4f})")

print("\nvoorbeeld uit het beste model:")
print(genereer(beste_model, tok, start="Op een dag ", n_nieuw=200, lengte=LENGTE,
               temperatuur=0.8))

torch.save({
    "state_dict": {k: v.cpu() for k, v in beste_model.state_dict().items()},
    "chars": tok.chars,
    "config": dict(n_embed=beste_inst["n_embed"], n_lagen=N_LAGEN, n_koppen=N_KOPPEN,
                   gebruik_positie=GEBRUIK_POSITIE, gebruik_feedforward=GEBRUIK_FEEDFORWARD,
                   losse_qk=LOSSE_QK, losse_v=LOSSE_V, gebruik_layernorm=GEBRUIK_LAYERNORM,
                   gebruik_masker=GEBRUIK_MASKER, uit_projectie=UIT_PROJECTIE, dropout=0.0,
                   max_lengte=LENGTE),
    "lengte": LENGTE,
}, MODEL_PAD)
print(f"\nmodel opgeslagen: {MODEL_PAD}")
