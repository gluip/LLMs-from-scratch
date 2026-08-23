"""Trekt dropout het overfit-gat van n_embed=160 dicht?

finale_vgl.py liet op de volle 18000 stappen zien dat n_embed=160, lr=5e-3
zwaar overfit: test 1,5874 tegen train 0,8870, een gat van 0,70. De standaard
(n_embed=80) haalde 1,3262 met een gat van maar 0,16. DROPOUT staat op 0,0 sinds
de dataset naar 1,8M karakters ging - dat is precies de knop tegen dit soort
overfitting, dus die zetten we hier terug aan voor het grote model.

Vraag: is er een dropout-waarde waarbij n_embed=160 de standaard verslaat, of
is 80 gewoon de betere match voor deze hoeveelheid data?
"""
import time
from pathlib import Path

import torch

from exp import (loss_per_positie, train_affiniteitsmodel, genereer, APPARAAT,
                 DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer,
                 N_LAGEN, N_KOPPEN, GEBRUIK_POSITIE, GEBRUIK_FEEDFORWARD,
                 LOSSE_QK, LOSSE_V, GEBRUIK_LAYERNORM, GEBRUIK_MASKER, UIT_PROJECTIE)

N_STAPPEN = 18000
LENGTE = 64
LR = 5e-3            # de beste lr voor n_embed=160 uit lr_vgl.py
N_EMBED = 160
DROPOUTS = [0.1, 0.2, 0.3]
MODEL_PAD = Path(__file__).parent / "model.pt"

# ankers uit eerdere runs, om de sweep in perspectief te zetten
STANDAARD_TEST = 1.3262     # n_embed=80, lr=1e-2, dropout=0.0, 18000 stappen
GROOT_ZONDER_DROPOUT = 1.5874  # n_embed=160, lr=5e-3, dropout=0.0, 18000 stappen

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
print(f"ankers: standaard test={STANDAARD_TEST:.4f}  "
      f"n_embed=160 zonder dropout test={GROOT_ZONDER_DROPOUT:.4f} (gat +0.70)\n", flush=True)

UITVOER = Path(__file__).parent / "dropout_resultaten.pt"
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
modellen = {}
for dropout in DROPOUTS:
    if dropout in resultaten:
        pp, eind_train = resultaten[dropout]
        print(f"dropout {dropout:.2f}  test {pp[-1]:.4f}  train {eind_train:.4f}  "
              f"gat {pp[-1]-eind_train:+.4f}  (overgeslagen)", flush=True)
        continue
    t0 = time.time()
    model, train_losses, *_ = train_affiniteitsmodel(
        n_lagen=N_LAGEN, n_embed=N_EMBED, lengte=LENGTE, dropout=dropout,
        gebruik_layernorm=True, lr=LR, n_stappen=N_STAPPEN,
        train_ids=train_ids, test_ids=test_ids, tokenizer=tok, eval_interval=3000)
    pp = loss_per_positie(model, test_ids, tok, LENGTE)
    eind_train = sum(train_losses[-100:]) / 100
    resultaten[dropout] = (pp, eind_train)
    modellen[dropout] = model
    print(f"\ndropout {dropout:.2f}  test {pp[-1]:.4f}  train {eind_train:.4f}  "
          f"gat {pp[-1]-eind_train:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
    torch.save(resultaten, UITVOER)

print("\n" + "=" * 70)
print(f"{'standaard':>14s}  test {STANDAARD_TEST:.4f}  (n_embed=80, referentie)")
print(f"{'160, drop 0.0':>14s}  test {GROOT_ZONDER_DROPOUT:.4f}  gat +0.7005")
for dropout in DROPOUTS:
    pp, eind_train = resultaten[dropout]
    print(f"{'160, drop ' + f'{dropout:.1f}':>14s}  test {pp[-1]:.4f}  train {eind_train:.4f}  "
          f"gat {pp[-1]-eind_train:+.4f}")

beste_dropout = min(DROPOUTS, key=lambda d: resultaten[d][0][-1].item())
beste_test = resultaten[beste_dropout][0][-1].item()
if beste_test < STANDAARD_TEST and beste_dropout in modellen:
    print(f"\nn_embed=160 met dropout={beste_dropout} wint ({beste_test:.4f} < {STANDAARD_TEST:.4f}) "
          f"-> wordt het nieuwe model.pt")
    beste_model = modellen[beste_dropout]
    print("\nvoorbeeld:")
    print(genereer(beste_model, tok, start="Op een dag ", n_nieuw=200, lengte=LENGTE, temperatuur=0.8))
    torch.save({
        "state_dict": {k: v.cpu() for k, v in beste_model.state_dict().items()},
        "chars": tok.chars,
        "config": dict(n_embed=N_EMBED, n_lagen=N_LAGEN, n_koppen=N_KOPPEN,
                       gebruik_positie=GEBRUIK_POSITIE, gebruik_feedforward=GEBRUIK_FEEDFORWARD,
                       losse_qk=LOSSE_QK, losse_v=LOSSE_V, gebruik_layernorm=GEBRUIK_LAYERNORM,
                       gebruik_masker=GEBRUIK_MASKER, uit_projectie=UIT_PROJECTIE, dropout=0.0,
                       max_lengte=LENGTE),
        "lengte": LENGTE,
    }, MODEL_PAD)
    print(f"model opgeslagen: {MODEL_PAD}")
else:
    print(f"\nbeste dropout-variant ({beste_test:.4f}) verslaat de standaard ({STANDAARD_TEST:.4f}) niet "
          f"-> model.pt blijft ongewijzigd")
