"""Vergelijk verschillende context-lengtes en plot de loss per positie."""
import time, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from exp import (maak_batch, train_affiniteitsmodel, DATA_MAP, TEKST_BESTANDEN,
                 TRAIN_FRACTIE, CharTokenizer)

N_STAPPEN = 5000
UITVOER = Path(__file__).parent / "venster_resultaten.pt"
VENSTERS = [16, 32, 64, 128]
# 256 is bewust weggelaten: 128 kostte al 2235s tegen 178s voor 64, en leverde
# op de laatste positie nog maar 0,012 op. 256 kost uren voor vrijwel niets.

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tok = CharTokenizer("".join(t for _, t in boeken))
tr, te = [], []
for _, t in boeken:
    d = torch.tensor(tok.encode(t), dtype=torch.long)
    s = int(TRAIN_FRACTIE * len(d))
    tr.append(d[:s]); te.append(d[s:])
train_ids, test_ids = torch.cat(tr), torch.cat(te)
print(f"vocab={tok.vocab_size} train={len(train_ids)}\n", flush=True)


def loss_per_positie(model, lengte, herhalingen=20):
    """Gemiddelde loss op elke positie in het venster.

    Positie 0 heeft nul karakters context, positie t heeft er t. Zo zie je
    rechtstreeks hoeveel elke extra karakter context nog waard is.
    """
    model.eval(); g = torch.Generator().manual_seed(1234)
    som = torch.zeros(lengte)
    for _ in range(herhalingen):
        x, y = maak_batch(test_ids, lengte, 256, g)
        with torch.no_grad():
            s, _ = model(x)
        pp = F.cross_entropy(s.reshape(-1, tok.vocab_size), y.reshape(-1),
                             reduction="none").view(y.shape)
        som += pp.mean(dim=0)
    model.train()
    return som / herhalingen


resultaten = {}
for lengte in VENSTERS:
    t0 = time.time()
    model, *_ = train_affiniteitsmodel(
        n_lagen=5, n_embed=80, lengte=lengte, dropout=0.0, gebruik_layernorm=True,
        n_stappen=N_STAPPEN, train_ids=train_ids, test_ids=test_ids, tokenizer=tok,
        eval_interval=10**9)
    pp = loss_per_positie(model, lengte)
    resultaten[lengte] = pp
    print(f"venster {lengte:>4d}  gemiddeld {pp.mean():.4f}  laatste positie {pp[-1]:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    # na elk venster wegschrijven: valt een latere run om, dan ben je de rest niet kwijt
    torch.save(resultaten, UITVOER)


