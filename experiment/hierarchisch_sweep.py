"""Sweep voor het hiërarchische model: kan een variant de char-baseline (1,2605
nats/karakter) verslaan? hierarchisch_vs_char_vgl.py liet zien dat de eerste
poging (1,3004) verliest - maar die leende lr=5e-3 en dropout=0,1 klakkeloos
van het char-model, zonder ze voor DEZE architectuur te verifiëren. Dat is
precies de fout uit EXPERIMENTEN.md experiment 3/4 (hyperparameters van de ene
architectuur op de andere plakken, ongetest).

Diagnose uit de trainingscurve van de eerste poging (hierarchisch_vgl.png):
het gat tussen train- en test-loss is klein (~0,1-0,15) - geen overfitting-
signatuur zoals bij het char-model zonder dropout, eerder een teken dat het
model nog niet tegen zijn plafond aanloopt. Drie gerichte varianten, elk een
VOLLE 18000-stappen-run (geen verkorte-run-valkuil zoals experiment 3):

  - "dropout 0"      : kleine gap => misschien is dropout hier pure kosten
  - "lr 8e-3"         : sneller/agressiever leren, misschien onderbenut nu
  - "grotere binnen"  : encoder/decoder hebben een moeilijkere taak (een heel
                        woord ontcijferen/opbouwen) dan het char-model's platte
                        opzoektabel - misschien is n_embed_binnen=64 een plafond
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

# --- vaste instellingen, gelijk aan de eerste hierarchische run ---
MAX_BROK_LENGTE = 16
BROK_VENSTER = 32
N_KOPPEN_BINNEN = 4
N_EMBED_BUITEN = 160
N_LAGEN_BUITEN = 5
N_KOPPEN_BUITEN = 4
AANTAL = 64
N_STAPPEN = 18000
EVAL_INTERVAL = 500

BASELINE_CHAR = 1.2605
EERSTE_POGING = 1.3004  # lr=5e-3, dropout=0.1, n_embed_binnen=64, n_lagen_enc/dec=2 - zie EXPERIMENTEN.md experiment 14

VARIANTEN = [
    ("baseline (lr5e-3,drop0.1,64/2)", dict(lr=5e-3, dropout=0.1, n_embed_binnen=64, n_lagen_enc=2, n_lagen_dec=2)),
    ("dropout 0",                       dict(lr=5e-3, dropout=0.0, n_embed_binnen=64, n_lagen_enc=2, n_lagen_dec=2)),
    ("lr 8e-3",                         dict(lr=8e-3, dropout=0.1, n_embed_binnen=64, n_lagen_enc=2, n_lagen_dec=2)),
    ("grotere binnen (96, 3 lagen)",    dict(lr=5e-3, dropout=0.1, n_embed_binnen=96, n_lagen_enc=3, n_lagen_dec=3)),
]

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tekst = "".join(t for _, t in boeken)
tok = CharTokenizer(tekst)
EOW_ID, PAD_ID = tok.vocab_size, tok.vocab_size + 1
print(f"apparaat={APPARAAT}  vocab={tok.vocab_size}  totaal={len(tekst):,} karakters\n", flush=True)

train_brokken, test_brokken = bouw_brok_tensor(boeken, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID, TRAIN_FRACTIE)
train_brokken, test_brokken = train_brokken.to(APPARAAT), test_brokken.to(APPARAAT)
print(f"brokken: train={train_brokken.shape[0]:,}  test={test_brokken.shape[0]:,}\n", flush=True)

UITVOER = Path(__file__).parent / "hierarchisch_sweep_resultaten.pt"
resultaten = torch.load(UITVOER) if UITVOER.exists() else {}

for naam, hp in VARIANTEN:
    if naam in resultaten:
        r = resultaten[naam]
        print(f"{naam:>32s}  nats/char {r['nats_excl']:.4f}  (overgeslagen)", flush=True)
        continue

    torch.manual_seed(0)
    model = HierarchischModel(
        tok_vocab_size=tok.vocab_size, pad_id=PAD_ID, eow_id=EOW_ID, max_brok_lengte=MAX_BROK_LENGTE,
        n_embed_binnen=hp["n_embed_binnen"], n_lagen_enc=hp["n_lagen_enc"], n_lagen_dec=hp["n_lagen_dec"],
        n_koppen_binnen=N_KOPPEN_BINNEN, n_embed_buiten=N_EMBED_BUITEN, n_lagen_buiten=N_LAGEN_BUITEN,
        n_koppen_buiten=N_KOPPEN_BUITEN, brok_venster=BROK_VENSTER, dropout=hp["dropout"],
    ).to(APPARAAT)
    n_par = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STAPPEN)
    char_vocab_out = model.decoder.uit.out_features

    print(f"\n--- {naam} ({n_par:,} parameters) ---", flush=True)
    t0 = time.time()
    train_losses, test_stappen, test_losses = [], [], []
    for stap in range(N_STAPPEN):
        x_b, y_b = maak_woord_batch(train_brokken, BROK_VENSTER, AANTAL)
        logits = model(x_b, y_b)
        loss = F.cross_entropy(logits.reshape(-1, char_vocab_out), y_b.reshape(-1), ignore_index=PAD_ID)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        train_losses.append(loss.item())

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

    # eerlijke nats/char (inhoud, EOW uitgesloten), 20 herhalingen zoals steeds
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
print(f"{'':>32s}  nats/char   verschil vs char-baseline")
print(f"{'char-model':>32s}  {BASELINE_CHAR:>9.4f}   (referentie)")
for naam, _ in VARIANTEN:
    r = resultaten[naam]
    print(f"{naam:>32s}  {r['nats_excl']:>9.4f}   {r['nats_excl']-BASELINE_CHAR:+.4f}")

beste_naam = min(resultaten, key=lambda n: resultaten[n]["nats_excl"])
beste = resultaten[beste_naam]
print(f"\nbeste variant: {beste_naam}  ({beste['nats_excl']:.4f})")
if beste["nats_excl"] < BASELINE_CHAR:
    print(f"WINT van de char-baseline! ({beste['nats_excl']:.4f} < {BASELINE_CHAR})")
else:
    print(f"wint nog niet van de char-baseline ({beste['nats_excl']:.4f} > {BASELINE_CHAR})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for (naam, _), kleur in zip(VARIANTEN, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    r = resultaten[naam]
    ax.plot(range(len(r["train_losses"])), r["train_losses"], color=kleur, alpha=0.2, linewidth=0.7)
    ax.plot(r["test_stappen"], r["test_losses"], color=kleur, marker="o", markersize=2.5, label=naam)
ax.axhline(BASELINE_CHAR, color="black", linestyle="--", label=f"char-baseline ({BASELINE_CHAR}, andere eenheid: excl. EOW)")
ax.set_xlabel("stap"); ax.set_ylabel("loss (nats/karakter incl. EOW, tijdens training)")
ax.set_title("hiërarchisch model: sweep over lr/dropout/binnen-capaciteit")
ax.legend(fontsize=8)
fig.tight_layout()
plot_pad = Path(__file__).parent / "hierarchisch_sweep.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
