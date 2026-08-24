"""Vervolg op hierarchisch_sweep.py: de winnende config was n_embed_binnen=96,
3 lagen, 18000 stappen (1,2139 nats/char, wint al van de char-baseline 1,2605).
Twee vragen die nog open stonden (zie EXPERIMENTEN.md experiment 16):

  1. Is 96/3 al een plafond, of gaat het patroon door? -> nog groter proberen
     (128 dim, 4 lagen).
  2. De loss-curve van de winnaar was bij stap 18000 nog niet duidelijk plat
     (train 0,854, test 0,877, nog dalend) - anders dan bij het char-model,
     waar langer_trainen.py liet zien dat de data toen al verzadigd was. Dit
     model doet per stap een moeilijkere taak, dus misschien is het nog niet
     uitgeleerd. -> dezelfde winnende config 36000 stappen laten draaien.

Beide een VOLLE run (geen verkorte-run-fout), tegen dezelfde vaste
instellingen als hierarchisch_sweep.py. De 18000-stappen-winnaar zelf hoeft
niet opnieuw: die staat al in hierarchisch_sweep_resultaten.pt.
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import APPARAAT, DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer
from hierarchisch import (splits_in_brokken, codeer_brok, bouw_brok_tensor, maak_woord_batch,
                          HierarchischModel, genereer_hierarchisch)

MAX_BROK_LENGTE = 16
BROK_VENSTER = 32
N_KOPPEN_BINNEN = 4
N_EMBED_BUITEN = 160
N_LAGEN_BUITEN = 5
N_KOPPEN_BUITEN = 4
AANTAL = 64
EVAL_INTERVAL = 500
LR, DROPOUT = 5e-3, 0.1

BASELINE_CHAR = 1.2605
WINNAAR_18K = 1.2139  # n_embed_binnen=96, 3 lagen, 18000 stappen - zie hierarchisch_sweep_resultaten.pt

VARIANTEN = [
    ("18k, 96/3 (winnaar - herbevestiging)", dict(n_embed_binnen=96, n_lagen_enc=3, n_lagen_dec=3, n_stappen=18000)),
    ("18k, 128/4 (nog groter)",              dict(n_embed_binnen=128, n_lagen_enc=4, n_lagen_dec=4, n_stappen=18000)),
    ("36k, 96/3 (langer trainen)",           dict(n_embed_binnen=96, n_lagen_enc=3, n_lagen_dec=3, n_stappen=36000)),
]

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tekst = "".join(t for _, t in boeken)
tok = CharTokenizer(tekst)
EOW_ID, PAD_ID = tok.vocab_size, tok.vocab_size + 1
print(f"apparaat={APPARAAT}  vocab={tok.vocab_size}  totaal={len(tekst):,} karakters\n", flush=True)

train_brokken, test_brokken = bouw_brok_tensor(boeken, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID, TRAIN_FRACTIE)
train_brokken, test_brokken = train_brokken.to(APPARAAT), test_brokken.to(APPARAAT)
print(f"brokken: train={train_brokken.shape[0]:,}  test={test_brokken.shape[0]:,}\n", flush=True)

# de bekende 18k/96-3-winnaar hergebruiken i.p.v. herdraaien, als sanity-check
oude_sweep = torch.load(Path(__file__).parent / "hierarchisch_sweep_resultaten.pt", weights_only=False)

UITVOER = Path(__file__).parent / "hierarchisch_sweep2_resultaten.pt"
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}
if "18k, 96/3 (winnaar - herbevestiging)" not in resultaten:
    r = oude_sweep["grotere binnen (96, 3 lagen)"]
    resultaten["18k, 96/3 (winnaar - herbevestiging)"] = dict(
        train_losses=r["train_losses"], test_stappen=r["test_stappen"], test_losses=r["test_losses"],
        nats_excl=r["nats_excl"], n_par=r["n_par"], voorbeeld=r["voorbeeld"], hergebruikt=True,
    )
    torch.save(resultaten, UITVOER)

for naam, hp in VARIANTEN:
    if naam in resultaten:
        r = resultaten[naam]
        print(f"{naam:>40s}  nats/char {r['nats_excl']:.4f}  "
              f"{'(hergebruikt uit sweep 1)' if r.get('hergebruikt') else '(overgeslagen)'}", flush=True)
        continue

    torch.manual_seed(0)
    model = HierarchischModel(
        tok_vocab_size=tok.vocab_size, pad_id=PAD_ID, eow_id=EOW_ID, max_brok_lengte=MAX_BROK_LENGTE,
        n_embed_binnen=hp["n_embed_binnen"], n_lagen_enc=hp["n_lagen_enc"], n_lagen_dec=hp["n_lagen_dec"],
        n_koppen_binnen=N_KOPPEN_BINNEN, n_embed_buiten=N_EMBED_BUITEN, n_lagen_buiten=N_LAGEN_BUITEN,
        n_koppen_buiten=N_KOPPEN_BUITEN, brok_venster=BROK_VENSTER, dropout=DROPOUT,
    ).to(APPARAAT)
    n_par = sum(p.numel() for p in model.parameters())
    n_stappen = hp["n_stappen"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_stappen)
    char_vocab_out = model.decoder.uit.out_features

    print(f"\n--- {naam} ({n_par:,} parameters, {n_stappen} stappen) ---", flush=True)
    t0 = time.time()
    train_losses, test_stappen, test_losses = [], [], []
    for stap in range(n_stappen):
        x_b, y_b = maak_woord_batch(train_brokken, BROK_VENSTER, AANTAL)
        logits = model(x_b, y_b)
        loss = F.cross_entropy(logits.reshape(-1, char_vocab_out), y_b.reshape(-1), ignore_index=PAD_ID)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        train_losses.append(loss.item())

        if stap % EVAL_INTERVAL == 0 or stap == n_stappen - 1:
            model.eval()
            with torch.no_grad():
                x_t, y_t = maak_woord_batch(test_brokken, BROK_VENSTER, AANTAL)
                logits_t = model(x_t, y_t)
                loss_t = F.cross_entropy(logits_t.reshape(-1, char_vocab_out), y_t.reshape(-1), ignore_index=PAD_ID)
            model.train()
            test_stappen.append(stap); test_losses.append(loss_t.item())
            print(f"  stap {stap:>6}  train {loss.item():.3f}  test {loss_t.item():.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    som_excl, aantal_excl = 0.0, 0
    with torch.no_grad():
        for _ in range(20):
            x_t, y_t = maak_woord_batch(test_brokken, BROK_VENSTER, 256)
            logits_t = model(x_t, y_t)
            verlies = F.cross_entropy(logits_t.reshape(-1, char_vocab_out), y_t.reshape(-1),
                                      ignore_index=PAD_ID, reduction="none")
            y_flat = y_t.reshape(-1)
            inhoud = (y_flat != PAD_ID) & (y_flat != EOW_ID)
            som_excl += verlies[inhoud].sum().item()
            aantal_excl += inhoud.sum().item()
    nats_excl = som_excl / aantal_excl

    voorbeeld_brokken = [codeer_brok(b, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID) for b in splits_in_brokken("Op een dag ")]
    voorbeeld_tekst, _ = genereer_hierarchisch(model, tok, voorbeeld_brokken, brok_venster=BROK_VENSTER,
                                               n_nieuwe_woorden=30, apparaat=APPARAAT, temperatuur=0.7)

    resultaten[naam] = dict(train_losses=train_losses, test_stappen=test_stappen, test_losses=test_losses,
                            nats_excl=nats_excl, n_par=n_par, voorbeeld=voorbeeld_tekst,
                            state_dict={k: v.cpu() for k, v in model.state_dict().items()}, hp=hp)
    print(f"  KLAAR: nats/karakter (excl EOW) = {nats_excl:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"  voorbeeld: {voorbeeld_tekst!r}", flush=True)
    torch.save(resultaten, UITVOER)

print("\n" + "=" * 70)
print(f"{'':>40s}  nats/char   verschil vs char-baseline")
print(f"{'char-model':>40s}  {BASELINE_CHAR:>9.4f}   (referentie)")
for naam, _ in [("18k, 96/3 (winnaar - herbevestiging)", None)] + VARIANTEN[1:]:
    r = resultaten[naam]
    print(f"{naam:>40s}  {r['nats_excl']:>9.4f}   {r['nats_excl']-BASELINE_CHAR:+.4f}")

alle_namen = ["18k, 96/3 (winnaar - herbevestiging)"] + [n for n, _ in VARIANTEN[1:]]
beste_naam = min(alle_namen, key=lambda n: resultaten[n]["nats_excl"])
beste = resultaten[beste_naam]
print(f"\nbeste variant: {beste_naam}  ({beste['nats_excl']:.4f})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for naam, kleur in zip(alle_namen, ["tab:blue", "tab:orange", "tab:green"]):
    r = resultaten[naam]
    ax.plot(range(len(r["train_losses"])), r["train_losses"], color=kleur, alpha=0.2, linewidth=0.7)
    ax.plot(r["test_stappen"], r["test_losses"], color=kleur, marker="o", markersize=2.5, label=naam)
ax.axhline(BASELINE_CHAR, color="black", linestyle="--", label=f"char-baseline ({BASELINE_CHAR}, andere eenheid: excl. EOW)")
ax.set_xlabel("stap"); ax.set_ylabel("loss (nats/karakter incl. EOW, tijdens training)")
ax.set_title("hiërarchisch model: nog groter, of langer trainen?")
ax.legend(fontsize=8)
fig.tight_layout()
plot_pad = Path(__file__).parent / "hierarchisch_sweep2.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
