# Kan het netwerk zijn eigen diepte kiezen, en helpt een grotere mutatie bij
# het splitsen van nodes?
#
# Twee losse eindjes uit experiment 13:
#
#  1. Daar werd de omvang van buitenaf bepaald: snoeien en groeien gebeurden
#     tussen de gradientstappen door, want "hoort deze node te bestaan" is
#     discreet en dus niet af te leiden. Dat kan anders. Zet een leerbare
#     poort alfa voor elk blok - h = h + alfa * blok(h) - met een straf op
#     |alfa|. Gaat alfa naar nul, dan is dat blok er effectief niet meer, en
#     de hele keuze loopt gewoon mee in backprop.
#
#  2. Het splitsen van nodes leverde niets op. Vermoedelijke oorzaak: twee
#     identieke kopieen krijgen identieke gradienten en blijven dus identiek.
#     Er stond 1% ruis op; dat is mogelijk te weinig om die symmetrie te
#     breken tegen weight decay in. Hier wordt de mutatiegrootte gevarieerd.
#
# Draaien:  .venv/bin/python -u exp-math/leerbare_vorm.py   (~25 min)

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import rekenen as r
import topologie as t

N_EMBED = 32
N_KOPPEN = 8
BREEDTE = 128
STAPPEN = 30000
SEEDS = range(3)
POORT_STRAF = 0.01      # straf op |alfa|, duwt ongebruikte blokken naar nul


class GepoortBlok(nn.Module):
    """Eén attentie- plus feedforward-blok, elk met een eigen leerbare poort.

    De poorten beginnen op 1 (blok volledig aan). De straf op |alfa| in de
    loss duwt ze omlaag; alleen blokken die hun bijdrage waard zijn blijven
    overeind. Zo bepaalt het netwerk zelf hoeveel lagen het gebruikt, zonder
    dat er iets van buitenaf verwijderd hoeft te worden.
    """

    def __init__(self, n_embed, n_koppen, breedte):
        super().__init__()
        self.n_embed, self.n_koppen = n_embed, n_koppen
        self.kop_dim = n_embed // n_koppen
        self.Q = nn.Linear(n_embed, n_embed, bias=False)
        self.K = nn.Linear(n_embed, n_embed, bias=False)
        self.V = nn.Linear(n_embed, n_embed, bias=False)
        self.W_o = nn.Linear(n_embed, n_embed)
        self.ff = nn.Sequential(nn.Linear(n_embed, breedte), nn.ReLU(),
                                nn.Linear(breedte, n_embed))
        self.poort_attentie = nn.Parameter(torch.ones(1))
        self.poort_ff = nn.Parameter(torch.ones(1))

    def _splits(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_koppen, self.kop_dim).transpose(1, 2)

    def forward(self, h):
        B, T, _ = h.shape
        q, k, v = self._splits(self.Q(h)), self._splits(self.K(h)), self._splits(self.V(h))
        aff = q @ k.transpose(-2, -1) / math.sqrt(self.kop_dim)
        meng = (torch.softmax(aff, -1) @ v).transpose(1, 2).contiguous().view(B, T, -1)
        h = h + self.poort_attentie * self.W_o(meng)
        h = h + self.poort_ff * self.ff(h)
        return h

    def poorten(self):
        return torch.cat([self.poort_attentie, self.poort_ff])


class GepoortModel(nn.Module):
    """Ruim bemeten stapel blokken; het netwerk mag zelf kiezen welke het houdt."""

    def __init__(self, vocab_size, n_blokken=4, n_embed=N_EMBED,
                 n_koppen=N_KOPPEN, breedte=BREEDTE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos = nn.Embedding(r.LENGTE, n_embed)
        self.blokken = nn.ModuleList([GepoortBlok(n_embed, n_koppen, breedte)
                                      for _ in range(n_blokken)])
        self.uit = nn.Linear(n_embed, 1)

    def forward(self, x):
        h = self.embed(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for blok in self.blokken:
            h = blok(h)
        return self.uit(h)[:, -1].squeeze(-1)

    def alle_poorten(self):
        return torch.stack([b.poorten() for b in self.blokken])


def train_gepoort(m, xtr, ytr, stappen=STAPPEN, seed=0, straf=POORT_STRAF):
    opt = torch.optim.AdamW(m.parameters(), lr=r.LEERRATE, weight_decay=r.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stappen)
    g = torch.Generator().manual_seed(seed)
    for _ in range(stappen):
        i = torch.randint(0, len(xtr), (r.BATCH_AANTAL,), generator=g)
        verlies = F.mse_loss(m(xtr[i]), ytr[i])
        if straf:
            verlies = verlies + straf * m.alle_poorten().abs().sum()
        opt.zero_grad(); verlies.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m


@torch.no_grad()
def meet(m, x, y):
    m.eval(); a = (m(x).round() == y).float().mean().item(); m.train(); return a


@torch.no_grad()
def groei_met_mutatie(model, masker, hoeveel, x, mutatie):
    """Als topologie.groei, maar met instelbare mutatiegrootte.

    Bij mutatie=0 zijn de twee helften exact gelijk. Ze krijgen dan ook exact
    dezelfde gradient, dus ze blijven voor altijd gelijk: de splitsing levert
    geen enkele nieuwe vrijheidsgraad op. Hoe groot de mutatie moet zijn om
    die symmetrie echt te breken is precies wat hier gemeten wordt.
    """
    leeg = (~masker).nonzero().squeeze(1)
    if len(leeg) == 0:
        return masker
    score = t.belang(model, x).clone()
    score[~masker] = -float("inf")
    druk = score.argsort(descending=True)[:min(hoeveel, len(leeg))]
    for bron, doel in zip(druk, leeg):
        w = model.ff[0].weight[bron]
        model.ff[0].weight[doel] = w * (1 + mutatie * torch.randn_like(w))
        model.ff[0].bias[doel] = model.ff[0].bias[bron]
        model.ff[2].weight[:, bron] /= 2
        model.ff[2].weight[:, doel] = model.ff[2].weight[:, bron]
        masker[doel] = True
    return masker


if __name__ == "__main__":
    tok, tw, xtr, ytr, xte, yte = r.laad("drie")

    print("DEEL 1 — laat het netwerk zijn eigen diepte kiezen\n")
    print("  4 blokken aangeboden, elk met een poort voor attentie en voor ff.")
    print(f"  Straf op |alfa|: {POORT_STRAF}. Een poort onder 0,1 tellen we als dicht.\n")
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = GepoortModel(tok.vocab_size)
        m = train_gepoort(m, xtr, ytr, seed=seed)
        p = m.alle_poorten().detach()
        open_ = (p.abs() > 0.1).sum().item()
        print(f"  seed {seed}: test {meet(m, xte, yte):>4.0%}   "
              f"{open_}/8 poorten open")
        for i in range(len(p)):
            merk = lambda v: "open " if abs(v) > 0.1 else "DICHT"
            print(f"    blok {i}: attentie {p[i,0]:+.3f} {merk(p[i,0])}   "
                  f"ff {p[i,1]:+.3f} {merk(p[i,1])}")

    print("\n  ter vergelijking, zonder straf op de poorten:")
    torch.manual_seed(0)
    m = train_gepoort(GepoortModel(tok.vocab_size), xtr, ytr, straf=0.0)
    p = m.alle_poorten().detach()
    print(f"    test {meet(m, xte, yte):.0%}, poorten "
          f"{[round(float(v),2) for v in p.flatten()]}")

    print("\n\nDEEL 2 — hoe groot moet de mutatie zijn bij het splitsen?\n")
    print(f"  {'mutatie':>9}{'voor':>8}{'direct na':>11}{'na bijschaven':>15}")
    for mutatie in (0.0, 0.01, 0.1, 0.5, 1.0):
        torch.manual_seed(0)
        m = r.Rekenmodel(tok.vocab_size, **t.BASIS)
        m.ff = nn.Sequential(nn.Linear(N_EMBED, BREEDTE), nn.ReLU(),
                             nn.Linear(BREEDTE, N_EMBED))
        masker = torch.zeros(BREEDTE, dtype=torch.bool); masker[:16] = True
        t.pas_masker_toe(m, masker)
        m = t.train_met_masker(m, xtr, ytr, masker, STAPPEN // 2)
        voor = meet(m, xte, yte)
        masker = groei_met_mutatie(m, masker, 16, xtr, mutatie)
        direct = meet(m, xte, yte)
        m = t.train_met_masker(m, xtr, ytr, masker, 10000, seed=42)
        print(f"  {mutatie:>9.2f}{voor:>8.0%}{direct:>11.0%}{meet(m, xte, yte):>15.0%}")
    print("\n  'direct na' hoort bij mutatie 0 gelijk te zijn aan 'voor':")
    print("  de splitsing is dan functie-behoudend. Bij grotere mutatie zakt hij")
    print("  eerst in, en de vraag is of het bijschaven dat meer dan goedmaakt.")
