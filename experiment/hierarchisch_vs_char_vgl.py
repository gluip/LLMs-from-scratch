"""Directe vergelijking: het hiërarchische chars->woord->transformer->chars
model (experiment 14) tegen het gewone char-model (experiment 13), beide al
getraind - geen hertraining, alleen laden, meten, en dezelfde prompts door
allebei laten genereren.

Eerlijke maat: nats/karakter. Voor het char-model is dat rechtstreeks
loss_per_positie's laatste positie. Voor het hiërarchische model is dat de
cross-entropy op de inhoudskarakters, EOW-events uitgesloten (zie
train_hierarchisch.py en EXPERIMENTEN.md experiment 14 voor waarom die twee
niet zomaar hetzelfde zijn - een char-model heeft geen EOW-concept).
"""
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import (APPARAAT, DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer,
                 AffiniteitsModel, maak_batch, loss_per_positie, genereer)
from hierarchisch import (HierarchischModel, splits_in_brokken, codeer_brok,
                          bouw_brok_tensor, maak_woord_batch, genereer_hierarchisch)

LENGTE_CHAR = 64          # trainingsvenster van het char-model, in karakters
TEMPERATUUR = 0.7
PROMPTS = ["Op een dag ", "Pinkeltje is ", "Het huis was "]
N_NIEUW_KARAKTERS = 150   # voor het char-model
N_NIEUW_WOORDEN = 30      # voor het hierarchische model (levert vergelijkbaar veel tekst op)

# ---------------------------------------------------------------------------
# Data: zelfde 14-boeken-corpus, zelfde train/test-split als beide trainingen
# ---------------------------------------------------------------------------
boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN]
tekst = "".join(t for _, t in boeken)
tok = CharTokenizer(tekst)
print(f"apparaat={APPARAAT}  vocab={tok.vocab_size}  totaal={len(tekst):,} karakters\n", flush=True)

tr, te = [], []
for _, t in boeken:
    d = torch.tensor(tok.encode(t), dtype=torch.long)
    s = int(TRAIN_FRACTIE * len(d))
    tr.append(d[:s]); te.append(d[s:])
train_ids, test_ids = torch.cat(tr).to(APPARAAT), torch.cat(te).to(APPARAAT)

# ---------------------------------------------------------------------------
# Char-model laden en meten
# ---------------------------------------------------------------------------
bundel_char = torch.load(Path(__file__).parent / "model.pt", weights_only=False, map_location="cpu")
model_char = AffiniteitsModel(tok.vocab_size, **bundel_char["config"])
model_char.load_state_dict(bundel_char["state_dict"])
model_char.eval().to(APPARAAT)
n_par_char = sum(p.numel() for p in model_char.parameters())

pp_char = loss_per_positie(model_char, test_ids, tok, LENGTE_CHAR)
nats_char = pp_char[-1].item()
print(f"char-model geladen: {n_par_char:,} parameters, nats/karakter (laatste positie) = {nats_char:.4f}", flush=True)

# ---------------------------------------------------------------------------
# Hierarchisch model laden en meten
# ---------------------------------------------------------------------------
bundel_hier = torch.load(Path(__file__).parent / "model_hierarchisch.pt", weights_only=False, map_location="cpu")
cfg = bundel_hier["config"]
model_hier = HierarchischModel(
    tok_vocab_size=cfg["tok_vocab_size"], pad_id=cfg["pad_id"], eow_id=cfg["eow_id"],
    max_brok_lengte=cfg["max_brok_lengte"], n_embed_binnen=cfg["n_embed_binnen"],
    n_lagen_enc=cfg["n_lagen_enc"], n_lagen_dec=cfg["n_lagen_dec"], n_koppen_binnen=cfg["n_koppen_binnen"],
    n_embed_buiten=cfg["n_embed_buiten"], n_lagen_buiten=cfg["n_lagen_buiten"],
    n_koppen_buiten=cfg["n_koppen_buiten"], brok_venster=cfg["brok_venster"], dropout=0.0,
)
model_hier.load_state_dict(bundel_hier["state_dict"])
model_hier.eval().to(APPARAAT)
n_par_hier = sum(p.numel() for p in model_hier.parameters())
BROK_VENSTER = cfg["brok_venster"]
EOW_ID, PAD_ID, M = cfg["eow_id"], cfg["pad_id"], cfg["max_brok_lengte"]

_, test_brokken = bouw_brok_tensor(boeken, tok, M, EOW_ID, PAD_ID, train_fractie=TRAIN_FRACTIE)
test_brokken = test_brokken.to(APPARAAT)
char_vocab_out = model_hier.decoder.uit.out_features

som_excl, aantal_excl = 0.0, 0
with torch.no_grad():
    for _ in range(20):
        x_t, y_t = maak_woord_batch(test_brokken, BROK_VENSTER, 256)
        logits_t = model_hier(x_t, y_t)
        verlies = F.cross_entropy(logits_t.reshape(-1, char_vocab_out), y_t.reshape(-1),
                                  ignore_index=PAD_ID, reduction="none")
        y_flat = y_t.reshape(-1)
        inhoud = (y_flat != PAD_ID) & (y_flat != EOW_ID)
        som_excl += verlies[inhoud].sum().item()
        aantal_excl += inhoud.sum().item()
nats_hier = som_excl / aantal_excl
print(f"hierarchisch model geladen: {n_par_hier:,} parameters, nats/karakter (inhoud, EOW uitgesloten) = {nats_hier:.4f}", flush=True)

# ---------------------------------------------------------------------------
# Samenvatting
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'':>28s}  nats/karakter   parameters")
print(f"{'char-model (model.pt)':>28s}  {nats_char:>13.4f}   {n_par_char:>11,}")
print(f"{'hierarchisch model':>28s}  {nats_hier:>13.4f}   {n_par_hier:>11,}")
print(f"verschil (hierarchisch - char): {nats_hier - nats_char:+.4f} nats/karakter")

# ---------------------------------------------------------------------------
# Zelfde prompts door allebei, naast elkaar
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GEGENEREERDE TEKST, ZELFDE PROMPTS")
for prompt in PROMPTS:
    print(f"\n>>> {prompt!r}")
    uit_char = genereer(model_char, tok, start=prompt, n_nieuw=N_NIEUW_KARAKTERS,
                        lengte=LENGTE_CHAR, temperatuur=TEMPERATUUR)
    print(f"  char        | {uit_char[len(prompt):]}")

    brokken_start = [codeer_brok(b, tok, M, EOW_ID, PAD_ID) for b in splits_in_brokken(prompt)]
    uit_hier, _ = genereer_hierarchisch(model_hier, tok, brokken_start, brok_venster=BROK_VENSTER,
                                        n_nieuwe_woorden=N_NIEUW_WOORDEN, apparaat=APPARAAT,
                                        temperatuur=TEMPERATUUR)
    print(f"  hierarchisch| {uit_hier[len(prompt):]}")

# ---------------------------------------------------------------------------
# Plot: nats/karakter naast elkaar (staafdiagram - de twee zijn niet op
# hetzelfde per-stap-meetmoment beschikbaar, dus geen samengevoegde curve;
# zie loss.png/beste_loss_14boeken.png en hierarchisch_vgl.png voor de losse
# trainingscurves van elk model apart)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
namen = ["char-model\n(model.pt)", "hiërarchisch\nmodel"]
waarden = [nats_char, nats_hier]
kleuren = ["tab:blue", "tab:orange"]
balken = ax.bar(namen, waarden, color=kleuren, width=0.5)
for balk, w in zip(balken, waarden):
    ax.annotate(f"{w:.4f}", (balk.get_x() + balk.get_width() / 2, w), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("nats per karakter (lager = beter)")
ax.set_title("char-model vs hiërarchisch model — eerlijke vergelijking")
ax.set_ylim(0, max(waarden) * 1.25)
fig.tight_layout()
plot_pad = Path(__file__).parent / "hierarchisch_vs_char_vgl.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
