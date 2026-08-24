"""Generaliseert RoPE naar een langer venster dan waarop getraind is?

rope_vgl.py trainde en evalueerde allebei de varianten op lengte=64 - een
eerlijke vergelijking, maar niet de test die er echt toe doet. Het beloofde
voordeel van RoPE is dat het geen vaste tabel per absolute positie gebruikt
(zoals gebruik_positie=True dat wel doet), maar Q/K roteert met een hoek die
voor élke positie uit te rekenen is - ook posities die nooit in een
trainingsvenster voorkwamen.

Dit script test dat letterlijk: beide modellen worden getraind op lengte=64
(zoals altijd), en daarna zonder hertraining geëvalueerd op lengte=128 - het
dubbele. Voor het RoPE-model betekent dat: gewoon de rope_cos/rope_sin-tabellen
verder doorrekenen (rope_hoeken is een pure functie van positie, geen geleerd
gewicht). Voor het absolute-positie-model betekent dat: pos_embed had nooit
rijen voor positie 64-127, dus die worden hier met verse, ongetrainde waarden
aangevuld - exact wat er zou gebeuren als genereer() zijn afkap-bescherming
niet had (zie de docstring daar).
"""
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp import (train_affiniteitsmodel, loss_per_positie, rope_hoeken, APPARAAT,
                 DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer, N_KOPPEN)

N_EMBED, LR, DROPOUT, LENGTE, N_STAPPEN = 160, 5e-3, 0.1, 64, 18000
LENGTE_LANG = 128  # het dubbele - nooit gezien tijdens training

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
      + f"  vocab={tok.vocab_size} train={len(train_ids):,}\n", flush=True)

VASTE_KWARGS = dict(
    n_lagen=5, n_embed=N_EMBED, lengte=LENGTE, dropout=DROPOUT, gebruik_layernorm=True,
    lr=LR, n_stappen=N_STAPPEN, n_koppen=N_KOPPEN, losse_qk=True, losse_v=True, uit_projectie=True,
    train_ids=train_ids, test_ids=test_ids, tokenizer=tok, eval_interval=10**9,
)

print("--- trainen zonder RoPE (absolute posities) ---", flush=True)
model_abs, *_ = train_affiniteitsmodel(gebruik_positie=True, gebruik_rope=False, **VASTE_KWARGS)

print("\n--- trainen met RoPE ---", flush=True)
model_rope, *_ = train_affiniteitsmodel(gebruik_positie=False, gebruik_rope=True, **VASTE_KWARGS)

# op lengte=64 (waar ze wel op getraind zijn) - moet dicht bij de bekende
# 1,2781 / 1,2605 uit rope_vgl.py uitkomen, puur als sanity-check dat dit
# dezelfde soort modellen zijn
pp_abs_64 = loss_per_positie(model_abs, test_ids, tok, LENGTE)
pp_rope_64 = loss_per_positie(model_rope, test_ids, tok, LENGTE)
print(f"\nop lengte=64 (getraind):  zonder RoPE {pp_abs_64[-1]:.4f}   met RoPE {pp_rope_64[-1]:.4f}", flush=True)

# --- uitbreiden naar lengte=128, zonder hertraining ---

# RoPE: de hoektabel is pure wiskunde, gewoon opnieuw uitrekenen tot 128
for laag in model_rope.lagen:
    cos, sin = rope_hoeken(laag.attentie.kop_dim, LENGTE_LANG)
    laag.attentie.rope_cos = cos.to(APPARAAT)
    laag.attentie.rope_sin = sin.to(APPARAAT)

# absolute posities: rijen 64-127 bestonden niet, dus nieuwe (ongetrainde) rijen
# aanplakken - precies wat er gebeurt als je voorbij max_lengte gaat zonder de
# afkap-bescherming die genereer() normaal gebruikt
oude_pos_embed = model_abs.pos_embed
nieuwe_pos_embed = nn.Embedding(LENGTE_LANG, N_EMBED).to(APPARAAT)
with torch.no_grad():
    nieuwe_pos_embed.weight[:LENGTE] = oude_pos_embed.weight  # de getrainde rijen behouden
    # rijen [LENGTE:] blijven op hun verse, ongetrainde initialisatie staan
model_abs.pos_embed = nieuwe_pos_embed

pp_abs_128 = loss_per_positie(model_abs, test_ids, tok, LENGTE_LANG)
pp_rope_128 = loss_per_positie(model_rope, test_ids, tok, LENGTE_LANG)
print(f"op lengte=128 (nooit gezien):  zonder RoPE {pp_abs_128[-1]:.4f}   met RoPE {pp_rope_128[-1]:.4f}", flush=True)

print("\n" + "=" * 70)
print(f"{'':>20s}  lengte=64 (getraind)   lengte=128 (extrapolatie)   verschil")
print(f"{'zonder RoPE':>20s}  {pp_abs_64[-1]:>18.4f}   {pp_abs_128[-1]:>22.4f}   {pp_abs_128[-1]-pp_abs_64[-1]:>+.4f}")
print(f"{'met RoPE':>20s}  {pp_rope_64[-1]:>18.4f}   {pp_rope_128[-1]:>22.4f}   {pp_rope_128[-1]-pp_rope_64[-1]:>+.4f}")

# ---------------------------------------------------------------------------
# Plot: loss per positie tot en met 128, met een lijn bij positie 64 waar de
# training-lengte ophield - daar zou zonder RoPE moeten omslaan
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
posities = range(1, LENGTE_LANG)
ax.plot(posities, pp_abs_128[1:], color="tab:blue", label="zonder RoPE (absolute posities)")
ax.plot(posities, pp_rope_128[1:], color="tab:orange", label="met RoPE")
ax.axvline(LENGTE, color="gray", linestyle="--", label=f"trainingsgrens (lengte={LENGTE})")
ax.set_xlabel("positie in het venster")
ax.set_ylabel("loss (nats)")
ax.set_title("loss per positie tot 2x de trainingslengte, zonder hertraining")
ax.legend()
fig.tight_layout()
plot_pad = Path(__file__).parent / "rope_extrapolatie.png"
fig.savefig(plot_pad)
print(f"\nplot opgeslagen: {plot_pad}", flush=True)
