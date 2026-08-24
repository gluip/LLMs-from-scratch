"""Vervolg op hierarchisch_sweep4.py: n_embed_buiten groter maken hielp niet
(experiment 20). Twee nieuwe assen, allebei nog nooit los getest voor deze
architectuur:

  1. Aantal lagen, los van breedte. Tot nu toe ging breedte en diepte aan de
     binnenkant altijd samen omhoog (64/2 -> 96/3 -> 128/4) - nooit getest of
     het de breedte was, de diepte, of allebei. Ook de buitenste diepte (5
     lagen, overgenomen van het char-model) is nog nooit los getest, net als
     n_embed_buiten dat niet was.
  2. FF_FACTOR (de fan-out/fan-in-factor van de feedforward-laag, nu 4.0
     overal, ongewijzigd sinds het begin van de hele sessie). Vereist een
     nieuwe, backward-compatible kwarg op Blok in exp.py (ff_factor=FF_FACTOR
     als default - verandert niets voor bestaande aanroepen, geverifieerd met
     een smoke-test).

Winnende config (n_embed_binnen=128, n_lagen_enc/dec=4, n_embed_buiten=160,
n_lagen_buiten=5, lr=3e-3) blijft het vertrekpunt; per variant wijzigt maar
1 ding.
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import APPARAAT, DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer, FF_FACTOR
from hierarchisch import (splits_in_brokken, codeer_brok, bouw_brok_tensor, maak_woord_batch,
                          HierarchischModel, genereer_hierarchisch)

MAX_BROK_LENGTE = 16
BROK_VENSTER = 32
N_KOPPEN_BINNEN = 4
N_KOPPEN_BUITEN = 4
AANTAL = 64
N_STAPPEN = 18000
EVAL_INTERVAL = 500
LR, DROPOUT = 3e-3, 0.1

BASELINE_CHAR = 1.2605
WINNAAR = 1.1730  # n_embed_binnen=128 4L, n_embed_buiten=160 5L, ff_factor=4.0 (default) - zie sweep3

# (naam, dict met overrides t.o.v. de winnende config)
STANDAARD = dict(n_embed_binnen=128, n_lagen_enc=4, n_lagen_dec=4,
                  n_embed_buiten=160, n_lagen_buiten=5, ff_factor_binnen=FF_FACTOR, ff_factor_buiten=FF_FACTOR)
VARIANTEN = [
    ("buiten 3 lagen (minder diep)",  dict(n_lagen_buiten=3)),
    ("buiten 8 lagen (dieper)",       dict(n_lagen_buiten=8)),
    ("binnen 2 lagen (minder diep, zelfde breedte 128)", dict(n_lagen_enc=2, n_lagen_dec=2)),
    ("binnen 6 lagen (dieper, zelfde breedte 128)",       dict(n_lagen_enc=6, n_lagen_dec=6)),
    ("ff_factor binnen 2.0 (smaller fan-out)", dict(ff_factor_binnen=2.0)),
    ("ff_factor binnen 8.0 (breder fan-out)",  dict(ff_factor_binnen=8.0)),
]

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tekst = "".join(t for _, t in boeken)
tok = CharTokenizer(tekst)
EOW_ID, PAD_ID = tok.vocab_size, tok.vocab_size + 1
print(f"apparaat={APPARAAT}  vocab={tok.vocab_size}  totaal={len(tekst):,} karakters\n", flush=True)

train_brokken, test_brokken = bouw_brok_tensor(boeken, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID, TRAIN_FRACTIE)
train_brokken, test_brokken = train_brokken.to(APPARAAT), test_brokken.to(APPARAAT)
print(f"brokken: train={train_brokken.shape[0]:,}  test={test_brokken.shape[0]:,}\n", flush=True)

UITVOER = Path(__file__).parent / "hierarchisch_sweep5_resultaten.pt"
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}

for naam, override in VARIANTEN:
    if naam in resultaten:
        r = resultaten[naam]
        print(f"{naam:>48s}  nats/char {r['nats_excl']:.4f}  (overgeslagen)", flush=True)
        continue

    hp = {**STANDAARD, **override}
    torch.manual_seed(0)
    model = HierarchischModel(
        tok_vocab_size=tok.vocab_size, pad_id=PAD_ID, eow_id=EOW_ID, max_brok_lengte=MAX_BROK_LENGTE,
        n_embed_binnen=hp["n_embed_binnen"], n_lagen_enc=hp["n_lagen_enc"], n_lagen_dec=hp["n_lagen_dec"],
        n_koppen_binnen=N_KOPPEN_BINNEN, n_embed_buiten=hp["n_embed_buiten"], n_lagen_buiten=hp["n_lagen_buiten"],
        n_koppen_buiten=N_KOPPEN_BUITEN, brok_venster=BROK_VENSTER, dropout=DROPOUT,
    ).to(APPARAAT)

    # ff_factor is geen HierarchischModel-kwarg (die bouwt zijn eigen Blok's intern
    # met de module-default FF_FACTOR) - voor deze sweep de binnenste Blok's er na
    # constructie op aanpassen, enige plek waar dat nu kan zonder hierarchisch.py
    # zelf een ff_factor-kwarg te geven (nog niet nodig voor 1 experiment)
    if hp["ff_factor_binnen"] != FF_FACTOR:
        from exp import FeedForwardLaag
        for laag in list(model.encoder.lagen) + list(model.decoder.lagen):
            if laag.feedforward is not None:
                laag.feedforward = FeedForwardLaag(hp["n_embed_binnen"], dropout=0.0,
                                                    factor=hp["ff_factor_binnen"]).to(APPARAAT)

    n_par = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STAPPEN)
    char_vocab_out = model.decoder.uit.out_features

    print(f"\n--- {naam} ({n_par:,} parameters) ---", flush=True)
    t0 = time.time()
    train_losses, test_stappen, test_losses = [], [], []
    hoogste_train_loss_na_opwarmen = 0.0
    for stap in range(N_STAPPEN):
        x_b, y_b = maak_woord_batch(train_brokken, BROK_VENSTER, AANTAL)
        logits = model(x_b, y_b)
        loss = F.cross_entropy(logits.reshape(-1, char_vocab_out), y_b.reshape(-1), ignore_index=PAD_ID)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        train_losses.append(loss.item())
        if stap > 1000:
            hoogste_train_loss_na_opwarmen = max(hoogste_train_loss_na_opwarmen, loss.item())

        if stap % EVAL_INTERVAL == 0 or stap == N_STAPPEN - 1:
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
                            hoogste_train_loss_na_opwarmen=hoogste_train_loss_na_opwarmen, hp=hp,
                            state_dict={k: v.cpu() for k, v in model.state_dict().items()})
    print(f"  KLAAR: nats/karakter (excl EOW) = {nats_excl:.4f}  "
          f"({'STABIEL' if hoogste_train_loss_na_opwarmen < 3 else 'INSTABIEL'})  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"  voorbeeld: {voorbeeld_tekst!r}", flush=True)
    torch.save(resultaten, UITVOER)

print("\n" + "=" * 70)
print(f"{'':>48s}  nats/char   verschil vs winnaar (1,1730)")
print(f"{'char-model':>48s}  {BASELINE_CHAR:>9.4f}")
print(f"{'huidige winnaar (128/4, buiten 160/5)':>48s}  {WINNAAR:>9.4f}   (referentie)")
for naam, _ in VARIANTEN:
    r = resultaten[naam]
    print(f"{naam:>48s}  {r['nats_excl']:>9.4f}   {r['nats_excl']-WINNAAR:>+9.4f}")

scores = {"huidige winnaar": WINNAAR}
for naam, _ in VARIANTEN:
    scores[naam] = resultaten[naam]["nats_excl"]
beste_naam = min(scores, key=scores.get)
print(f"\nbeste: {beste_naam}  ({scores[beste_naam]:.4f})")
if beste_naam != "huidige winnaar" and scores[beste_naam] < WINNAAR:
    print("NIEUWE WINNAAR gevonden!")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))
kleuren = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
for (naam, _), kleur in zip(VARIANTEN, kleuren):
    r = resultaten[naam]
    ax.plot(range(len(r["train_losses"])), r["train_losses"], color=kleur, alpha=0.15, linewidth=0.6)
    ax.plot(r["test_stappen"], r["test_losses"], color=kleur, marker="o", markersize=2, label=naam)
ax.axhline(WINNAAR, color="black", linestyle=":", label=f"huidige winnaar eindresultaat ({WINNAAR})")
ax.set_ylim(0, 3)
ax.set_xlabel("stap"); ax.set_ylabel("loss (nats/karakter incl. EOW, tijdens training)")
ax.set_title("hiërarchisch model: lagen (buiten/binnen) en ff_factor")
ax.legend(fontsize=7)
fig.tight_layout()
plot_pad = Path(__file__).parent / "hierarchisch_sweep5.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
