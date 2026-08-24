"""Volle, vergelijkbare run van het hiërarchische chars->woord->transformer->chars
model (zie hierarchisch.py en ontwerp_emergente_woordlaag.html), op de schone
14-boeken-dataset, tegen de hyperparameters van het huidige beste char-model
(n_embed=160, dropout=0,1, lr=5e-3, gebruik_rope=True, 5 buitenste lagen).

Twee dingen die de smoke-test (hierarchisch.py) niet deed en dit script wel:
1. eerst de tijd/stap meten op volle schaal, voordat n_stappen wordt vastgezet
   (de smoke-test was te klein om daar iets zinnigs over te zeggen);
2. de eerlijke nats/karakter-vergelijking tegen de bekende 1,2605-baseline,
   in twee getallen (met en zonder EOW-events meegeteld), zoals het plan
   voorschrijft - een char-model heeft geen EOW-concept, dus die twee zijn
   niet zomaar hetzelfde.
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
                          HierarchischModel, train_hierarchisch, genereer_hierarchisch)

# --- hyperparameters: volle run uit het plan ---
MAX_BROK_LENGTE = 16
BROK_VENSTER = 32          # ~64 karakters, vergelijkbaar met LENGTE=64 in exp.py
N_EMBED_BINNEN = 64
N_LAGEN_ENC = 2
N_LAGEN_DEC = 2
N_KOPPEN_BINNEN = 4
N_EMBED_BUITEN = 160        # matcht het huidige beste char-model
N_LAGEN_BUITEN = 5
N_KOPPEN_BUITEN = 4
DROPOUT = 0.1
LR = 5e-3
AANTAL = 64                 # batchgrootte, matcht BATCH_AANTAL in exp.py
EVAL_INTERVAL = 200

BASELINE_LAATSTE_POSITIE = 1.2605  # huidig beste char-model (model.pt), zie EXPERIMENTEN.md

boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tekst = "".join(t for _, t in boeken)
tok = CharTokenizer(tekst)
print(f"vocab={tok.vocab_size}  totaal={len(tekst):,} karakters\n", flush=True)

EOW_ID = tok.vocab_size       # geldige uitvoerklasse (0..vocab_size)
PAD_ID = tok.vocab_size + 1   # alleen als input, nooit voorspeld (ignore_index)

t0 = time.time()
train_brokken, test_brokken = bouw_brok_tensor(boeken, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID,
                                                train_fractie=TRAIN_FRACTIE)
print(f"brok-tensors gebouwd in {time.time()-t0:.0f}s: train={train_brokken.shape[0]:,}  "
      f"test={test_brokken.shape[0]:,}  (M={MAX_BROK_LENGTE})", flush=True)

torch.manual_seed(0)
model = HierarchischModel(
    tok_vocab_size=tok.vocab_size, pad_id=PAD_ID, eow_id=EOW_ID, max_brok_lengte=MAX_BROK_LENGTE,
    n_embed_binnen=N_EMBED_BINNEN, n_lagen_enc=N_LAGEN_ENC, n_lagen_dec=N_LAGEN_DEC,
    n_koppen_binnen=N_KOPPEN_BINNEN, n_embed_buiten=N_EMBED_BUITEN, n_lagen_buiten=N_LAGEN_BUITEN,
    n_koppen_buiten=N_KOPPEN_BUITEN, brok_venster=BROK_VENSTER, dropout=DROPOUT,
).to(APPARAAT)
n_par = sum(p.numel() for p in model.parameters())
print(f"model: {n_par:,} parameters\n", flush=True)

# --- tijd/stap kalibreren voordat n_stappen wordt vastgezet ---
train_brokken_dev = train_brokken.to(APPARAAT)
test_brokken_dev = test_brokken.to(APPARAAT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
char_vocab_out = model.decoder.uit.out_features

torch.cuda.synchronize() if APPARAAT.type == "cuda" else None
t0 = time.time()
KALIBRATIE_STAPPEN = 100
for _ in range(KALIBRATIE_STAPPEN):
    x_b, y_b = maak_woord_batch(train_brokken_dev, BROK_VENSTER, AANTAL)
    logits = model(x_b, y_b)
    loss = F.cross_entropy(logits.reshape(-1, char_vocab_out), y_b.reshape(-1), ignore_index=PAD_ID)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
torch.cuda.synchronize() if APPARAAT.type == "cuda" else None
ms_per_stap = 1000 * (time.time() - t0) / KALIBRATIE_STAPPEN
print(f"kalibratie: {ms_per_stap:.1f}ms/stap (gemeten over {KALIBRATIE_STAPPEN} stappen op volle schaal)")
for kandidaat in (2000, 5000, 9000, 18000):
    print(f"  {kandidaat:>6} stappen zou ~{kandidaat*ms_per_stap/1000/60:.1f} minuten kosten")

# kies n_stappen: zoveel mogelijk richting 18000 (consistent met de rest van het
# project), maar begrens de wandkloktijd - dit is de eerste volle run, geen zin
# om blind een uur te draaien als de kalibratie iets anders leert dan verwacht
BUDGET_MINUTEN = 20
N_STAPPEN = min(18000, int(BUDGET_MINUTEN * 60 * 1000 / ms_per_stap))
N_STAPPEN = max(N_STAPPEN, 2000)  # ondergrens, anders is de run zinloos
print(f"\ngekozen: n_stappen={N_STAPPEN} (budget ~{BUDGET_MINUTEN} min, geschat {N_STAPPEN*ms_per_stap/1000/60:.1f} min)\n", flush=True)

# --- de echte training (de kalibratiestappen hierboven tellen al mee als opwarming/vooruitgang) ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STAPPEN)
train_losses, test_stappen, test_losses = [], [], []
t0 = time.time()
for stap in range(N_STAPPEN):
    x_b, y_b = maak_woord_batch(train_brokken_dev, BROK_VENSTER, AANTAL)
    logits = model(x_b, y_b)
    loss = F.cross_entropy(logits.reshape(-1, char_vocab_out), y_b.reshape(-1), ignore_index=PAD_ID)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    train_losses.append(loss.item())

    if stap % EVAL_INTERVAL == 0 or stap == N_STAPPEN - 1:
        model.eval()
        with torch.no_grad():
            x_t, y_t = maak_woord_batch(test_brokken_dev, BROK_VENSTER, AANTAL)
            logits_t = model(x_t, y_t)
            loss_t = F.cross_entropy(logits_t.reshape(-1, char_vocab_out), y_t.reshape(-1), ignore_index=PAD_ID)
        model.train()
        test_stappen.append(stap); test_losses.append(loss_t.item())
        huidige_lr = optimizer.param_groups[0]["lr"]
        verstreken = time.time() - t0
        print(f"  stap {stap:>6}  train loss {loss.item():.3f}  test loss {loss_t.item():.3f}  "
              f"lr {huidige_lr:.5f}  ({verstreken:.0f}s)", flush=True)

print(f"\ntraining klaar in {(time.time()-t0)/60:.1f} minuten", flush=True)

# ---------------------------------------------------------------------------
# Eerlijke vergelijking: nats per karakter, met en zonder EOW-events
# ---------------------------------------------------------------------------
model.eval()
HERHALINGEN = 20
som_incl, aantal_incl = 0.0, 0
som_excl, aantal_excl = 0.0, 0
with torch.no_grad():
    for _ in range(HERHALINGEN):
        x_t, y_t = maak_woord_batch(test_brokken_dev, BROK_VENSTER, 256)
        logits_t = model(x_t, y_t)
        verlies_per_positie = F.cross_entropy(
            logits_t.reshape(-1, char_vocab_out), y_t.reshape(-1),
            ignore_index=PAD_ID, reduction="none"
        ).view(y_t.shape)  # (256, BROK_VENSTER, M)
        y_flat = y_t.reshape(-1)
        verlies_flat = verlies_per_positie.reshape(-1)
        geldig = y_flat != PAD_ID
        som_incl += verlies_flat[geldig].sum().item()
        aantal_incl += geldig.sum().item()
        inhoud = geldig & (y_flat != EOW_ID)
        som_excl += verlies_flat[inhoud].sum().item()
        aantal_excl += inhoud.sum().item()

nats_incl = som_incl / aantal_incl   # inhoud + EOW-events
nats_excl = som_excl / aantal_excl   # alleen inhoudskarakters, eerlijk vergelijkbaar met de baseline

print("\n" + "=" * 70)
print(f"nats/karakter (inhoud, EOW uitgesloten):     {nats_excl:.4f}   <- eerlijk vergelijkbaar met baseline")
print(f"nats/(karakter + EOW-events):                {nats_incl:.4f}   <- het echte trainingsdoel van dit model")
print(f"baseline (char-model, model.pt):              {BASELINE_LAATSTE_POSITIE:.4f}")
print(f"verschil (excl. EOW) t.o.v. baseline:         {nats_excl - BASELINE_LAATSTE_POSITIE:+.4f}")

# ---------------------------------------------------------------------------
# Validatie: buurwoorden, genereren, lengteverdeling, percentage bestaande woorden
# ---------------------------------------------------------------------------
print("\n--- validatie: buurwoorden ---")
paren = [("huis", "huizen"), ("huis", "visser"), ("speelde", "speelden"),
         ("de", "het"), ("mooi", "mooie"), ("Pinkeltje", "kabouter")]
woorden_flat = sorted(set(w for p in paren for w in p))
with torch.no_grad():
    ids = torch.tensor([codeer_brok(w, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID) for w in woorden_flat], device=APPARAAT)
    vecs = model.encoder(ids)
vec_van = dict(zip(woorden_flat, vecs))
for a, b in paren:
    sim = F.cosine_similarity(vec_van[a].unsqueeze(0), vec_van[b].unsqueeze(0)).item()
    print(f"  cos({a!r}, {b!r}) = {sim:+.3f}")

print("\n--- validatie: genereren ---")
start_brokken = [codeer_brok(b, tok, MAX_BROK_LENGTE, EOW_ID, PAD_ID) for b in splits_in_brokken("Op een dag ")]
volledige_tekst, nieuwe_woorden = genereer_hierarchisch(
    model, tok, start_brokken, brok_venster=BROK_VENSTER, n_nieuwe_woorden=60,
    apparaat=APPARAAT, temperatuur=0.8)
print(f"  {volledige_tekst!r}")

alle_brokken_corpus = splits_in_brokken(tekst)
echte_woorden = set(b for b in alle_brokken_corpus if b.strip())
echte_lengtes = sorted(len(b) for b in echte_woorden)
gen_woorden = [w for w in nieuwe_woorden if w.strip()]
gen_lengtes = sorted(len(w) for w in gen_woorden) if gen_woorden else [0]
print(f"  woordlengte mediaan: echt={echte_lengtes[len(echte_lengtes)//2]}  "
      f"gegenereerd={gen_lengtes[len(gen_lengtes)//2]}")
aantal_bestaand = sum(1 for w in gen_woorden if w in echte_woorden)
print(f"  percentage bestaande woorden: {aantal_bestaand}/{len(gen_woorden)} "
      f"({100*aantal_bestaand/max(1,len(gen_woorden)):.0f}%)")

# ---------------------------------------------------------------------------
# Plot en opslaan
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.plot(range(len(train_losses)), train_losses, alpha=0.3, label="train")
plt.plot(test_stappen, test_losses, marker="o", markersize=3, label="test")
plt.axhline(BASELINE_LAATSTE_POSITIE, color="gray", linestyle="--", label=f"char-baseline ({BASELINE_LAATSTE_POSITIE})")
plt.xlabel("stap"); plt.ylabel("loss (nats/karakter incl. EOW)")
plt.title(f"hiërarchisch model: train/test loss (n_embed_buiten={N_EMBED_BUITEN}, venster={BROK_VENSTER} brokken)")
plt.legend(); plt.tight_layout()
plot_pad = Path(__file__).parent / "hierarchisch_vgl.png"
plt.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)

torch.save({
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "chars": tok.chars,
    "config": dict(tok_vocab_size=tok.vocab_size, pad_id=PAD_ID, eow_id=EOW_ID,
                   max_brok_lengte=MAX_BROK_LENGTE, n_embed_binnen=N_EMBED_BINNEN,
                   n_lagen_enc=N_LAGEN_ENC, n_lagen_dec=N_LAGEN_DEC, n_koppen_binnen=N_KOPPEN_BINNEN,
                   n_embed_buiten=N_EMBED_BUITEN, n_lagen_buiten=N_LAGEN_BUITEN,
                   n_koppen_buiten=N_KOPPEN_BUITEN, brok_venster=BROK_VENSTER, dropout=0.0),
    "brok_venster": BROK_VENSTER,
}, Path(__file__).parent / "model_hierarchisch.pt")
print(f"model opgeslagen: model_hierarchisch.pt", flush=True)
