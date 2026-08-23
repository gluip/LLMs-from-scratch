"""Vergelijk verschillende context-lengtes en plot de loss per positie."""
import time, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from exp import (loss_per_positie, train_affiniteitsmodel, APPARAAT, DATA_MAP,
                 TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer)

N_STAPPEN = 5000
UITVOER = Path(__file__).parent / "venster_resultaten.pt"
VENSTERS = [16, 32, 64, 128]
# 256 was weggelaten toen dit nog op de CPU draaide: 128 kostte daar 2235s tegen
# 178s voor 64. Op de GPU is dat 69s tegen 49s, dus de rekentijd is geen argument
# meer. Het inhoudelijke argument staat wel: 64 -> 128 leverde op de laatste
# positie nog maar 0,005 op, dus van 256 valt weinig te verwachten.

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



# al eerder berekende vensters overslaan: trainen kost minuten, plotten seconden
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
for lengte in VENSTERS:
    if lengte in resultaten:
        pp = resultaten[lengte]
        print(f"venster {lengte:>4d}  gemiddeld {pp.mean():.4f}  laatste positie {pp[-1]:.4f}  "
              f"(overgeslagen, stond al in {UITVOER.name})", flush=True)
        continue
    t0 = time.time()
    model, *_ = train_affiniteitsmodel(
        n_lagen=5, n_embed=80, lengte=lengte, dropout=0.0, gebruik_layernorm=True,
        n_stappen=N_STAPPEN, train_ids=train_ids, test_ids=test_ids, tokenizer=tok,
        eval_interval=10**9)
    pp = loss_per_positie(model, test_ids, tok, lengte)
    resultaten[lengte] = pp
    print(f"venster {lengte:>4d}  gemiddeld {pp.mean():.4f}  laatste positie {pp[-1]:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    # na elk venster wegschrijven: valt een latere run om, dan ben je de rest niet kwijt
    torch.save(resultaten, UITVOER)


# ---------------------------------------------------------------------------
# Plot: links het verloop binnen een venster, rechts de samenvatting per venster
# ---------------------------------------------------------------------------
vensters = sorted(resultaten)
kleuren = plt.cm.viridis([i / max(1, len(vensters) - 1) for i in range(len(vensters))])
fig, (links, rechts) = plt.subplots(1, 2, figsize=(13, 5))

def glad(waarden, venster=9):
    """Voortschrijdend gemiddelde, zodat de trend door de meetruis heen zichtbaar is.

    De late posities zijn ruizig: elke positie is maar over 20x256 stukjes
    gemiddeld, en dat is te weinig om verschillen van 0,01 nats uit elkaar te
    houden. De ruwe lijn blijft er licht achter staan, zodat je die ruis ziet
    in plaats van dat het gladstrijken hem verstopt.
    """
    uit = torch.empty_like(waarden)
    for i in range(len(waarden)):
        a, b = max(0, i - venster // 2), min(len(waarden), i + venster // 2 + 1)
        uit[i] = waarden[a:b].mean()
    return uit


for kleur, lengte in zip(kleuren, vensters):
    pp = resultaten[lengte]
    # positie 0 valt weg: die heeft nul karakters context en zit rond de 4 nats,
    # dat zou de rest van de grafiek platdrukken. Op de log-as begint hij toch niet.
    posities = range(1, lengte)
    links.plot(posities, pp[1:], color=kleur, alpha=0.25, linewidth=0.8)
    links.plot(posities, glad(pp[1:]), color=kleur, linewidth=2, label=f"venster {lengte}")
links.set_xscale("log")
links.set_xlabel("positie in het venster (= aantal karakters context)")
links.set_ylabel("loss (nats)")
links.set_title("loss per positie: wat levert elk extra karakter context op?")
links.legend()
links.grid(alpha=0.3)

gemiddeld = [resultaten[l].mean().item() for l in vensters]
laatste = [resultaten[l][-1].item() for l in vensters]
rechts.plot(vensters, gemiddeld, marker="o", label="gemiddeld over het venster")
rechts.plot(vensters, laatste, marker="s", label="op de laatste positie")
rechts.set_xscale("log", base=2)
rechts.set_xticks(vensters); rechts.set_xticklabels(vensters)
rechts.set_xlabel("venstergrootte")
rechts.set_ylabel("loss (nats)")
# het gemiddelde daalt harder dan de laatste positie, en dat is deels een
# meetartefact: een groot venster heeft naar verhouding minder van die dure
# beginposities zonder context. De rechterlijn is de eerlijke vergelijking.
rechts.set_title("gemiddelde vs. laatste positie per venster")
rechts.legend()
rechts.grid(alpha=0.3)

fig.tight_layout()
plot_pad = Path(__file__).parent / "venster_vgl.png"
fig.savefig(plot_pad, dpi=130)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)


