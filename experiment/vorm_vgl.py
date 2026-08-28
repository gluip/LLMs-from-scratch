"""Topologie en activatiefuncties leren, toegepast op het taalmodel.

In exp-math is uitgezocht of een netwerk zijn eigen vorm kan leren: nodes die
weinig doen weghalen, drukke nodes splitsen, en elke node zijn eigen
activatiefunctie laten kiezen. Daar bleek van alles, maar met één groot
voorbehoud: de rekentaak was te makkelijk. Vier tot tweeëndertig verborgen
eenheden gaven allemaal hetzelfde resultaat, dus er viel niets te kiezen.

Dit script haalt dezelfde vragen naar een taak waar het model wél tegen zijn
capaciteitsgrens zit: 10,2M karakters Nederlands met 417k parameters.

Twee vragen:

  1. GEBRUIK-OF-VERLIES. Houd een vast budget aan actieve feedforward-eenheden.
     Gooi elke zoveel stappen de zwakste eruit en zet er evenveel WILLEKEURIGE
     nieuwe voor in de plaats. Draagt een nieuwe eenheid niets bij, dan is hij
     de volgende ronde weer de zwakste en verdwijnt hij vanzelf - de training
     zelf is de selectie, er is geen aparte evaluatie per kandidaat nodig.
     (In de kern SET/RigL.) Vergeleken met een vast willekeurig deel en met
     een even groot dicht model.

  2. ACTIVATIEKEUZE. Bij het rekenmodel koos het netwerk massaal het kwadraat,
     want a*b = ((a+b)^2-(a-b)^2)/4. Wat kiest het bij taal? Elke eenheid
     krijgt logits over vier functies; een softmax daarover geeft de
     mengverhouding, en die logits trainen gewoon mee.

Let op: dit draait op 6000 stappen, niet de 18000 van het beste model, om
meerdere condities te kunnen vergelijken. De absolute losses zijn dus hoger
dan in de tabel bovenaan EXPERIMENTEN.md; alleen de onderlinge vergelijking
telt.

Draaien:  .venv/bin/python -u experiment/vorm_vgl.py    (~50 min op CPU)
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from exp import (APPARAAT, DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE, CharTokenizer,
                 AffiniteitsModel, maak_batch, loss_per_positie)

N_EMBED, N_LAGEN, N_KOPPEN, LENGTE = 80, 5, 4, 64
N_STAPPEN, BATCH, LR = 6000, 64, 3e-3
# Op CPU kost één run ~4 minuten en is 6000 stappen een compromis. Op een GPU
# hoort dit op 18000 te staan (zoals het beste model in EXPERIMENTEN.md) en
# SEEDS op range(3) of meer; de conclusies hieronder zijn met 6000/2 gemeten.
FF_BREEDTE = int(N_EMBED * 4)      # zoals FF_FACTOR in exp.py
BUDGET = FF_BREEDTE // 4           # kwart actief
INTERVAL, VERVERS = 500, 0.20
SEEDS = range(2)

FUNCTIES = {
    "relu":     torch.relu,
    "gelu":     F.gelu,
    "kwadraat": lambda x: x * x / 2,
    "lineair":  lambda x: x,
}


def laad_data():
    tekst = "".join((DATA_MAP / n).read_text(encoding="utf-8") for n in TEKST_BESTANDEN)
    tok = CharTokenizer(tekst)
    ids = torch.tensor(tok.encode(tekst), dtype=torch.long)
    knip = int(len(ids) * TRAIN_FRACTIE)
    return tok, ids[:knip].to(APPARAAT), ids[knip:].to(APPARAAT)


def bouw(vocab, seed=0):
    torch.manual_seed(seed)
    return AffiniteitsModel(
        vocab, n_embed=N_EMBED, n_lagen=N_LAGEN, gebruik_positie=True,
        gebruik_feedforward=True, losse_qk=True, losse_v=True,
        gebruik_layernorm=True, n_koppen=N_KOPPEN, uit_projectie=True,
        max_lengte=LENGTE).to(APPARAAT)


def ff_lagen(model):
    """De twee Linears van elke feedforward, per laag."""
    return [(b.feedforward.net[0], b.feedforward.net[2]) for b in model.lagen]


@torch.no_grad()
def pas_maskers_toe(model, maskers):
    for (eerste, tweede), masker in zip(ff_lagen(model), maskers):
        eerste.weight[~masker] = 0
        eerste.bias[~masker] = 0
        tweede.weight[:, ~masker] = 0


@torch.no_grad()
def belang(model, x):
    """Per feedforward-eenheid: spreiding van de activatie x grootte uitgaand.

    Een eenheid die stilstaat draagt niets bij; een die schommelt maar nergens
    naartoe gaat ook niet. Dit vraagt één forward pass en is dus verwaarloosbaar
    naast de training.
    """
    scores = []
    h = model.embed(x)
    if model.gebruik_positie:
        h = h + model.pos_embed(torch.arange(x.shape[1], device=x.device))
    for blok in model.lagen:
        hn = blok.ln1(h) if blok.ln1 is not None else h
        uit, _ = blok.attentie(hn)
        h = h + uit
        hf = blok.ln2(h) if blok.ln2 is not None else h
        verborgen = torch.relu(blok.feedforward.net[0](hf))
        scores.append(verborgen.std((0, 1)) * blok.feedforward.net[2].weight.abs().mean(0))
        h = h + blok.feedforward(hf)
    return scores


@torch.no_grad()
def zet_aan(model, maskers, keuzes, schaal=0.05):
    """Wek plekken op met kleine willekeurige gewichten.

    Uitgaand mag niet exact nul zijn: de gradient op de ingaande gewichten is
    evenredig met de uitgaande, dus een eenheid die op nul begint krijgt nooit
    een duw en blijft liggen.
    """
    for (eerste, tweede), masker, plekken in zip(ff_lagen(model), maskers, keuzes):
        for p in plekken:
            eerste.weight[p] = torch.randn_like(eerste.weight[p]) * schaal
            eerste.bias[p] = 0
            tweede.weight[:, p] = torch.randn_like(tweede.weight[:, p]) * schaal
            masker[p] = True
    return maskers


@torch.no_grad()
def ververs(model, maskers, x, aantal):
    """Zwakste `aantal` per laag eruit, evenveel willekeurige nieuwe erin."""
    scores = belang(model, x)
    keuzes = []
    for score, masker in zip(scores, maskers):
        s = score.clone(); s[~masker] = float("inf")
        masker[s.argsort()[:aantal]] = False
    pas_maskers_toe(model, maskers)
    for masker in maskers:
        leeg = (~masker).nonzero().squeeze(1)
        # randperm op hetzelfde apparaat als `leeg`: een CUDA-tensor indexeren
        # met CPU-indices werkt niet in elke torch-versie
        volgorde = torch.randperm(len(leeg), device=leeg.device)
        keuzes.append(leeg[volgorde[:aantal]])
    return zet_aan(model, maskers, keuzes)


def train(model, train_ids, maskers=None, verversen=False, seed=0, stappen=N_STAPPEN):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stappen)
    g = torch.Generator().manual_seed(seed)
    for stap in range(stappen):
        x, y = maak_batch(train_ids, LENGTE, BATCH, g)
        s, _ = model(x)
        verlies = F.cross_entropy(s.reshape(-1, s.shape[-1]), y.reshape(-1))
        opt.zero_grad(); verlies.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if maskers is not None:
            pas_maskers_toe(model, maskers)
            if verversen and stap and stap % INTERVAL == 0 and stap < stappen * 0.8:
                ververs(model, maskers, x, max(1, int(BUDGET * VERVERS)))
    return model


def eind_loss(model, test_ids, tok):
    """Loss op de laatste positie — de maat die EXPERIMENTEN.md gebruikt."""
    return loss_per_positie(model, test_ids, tok, LENGTE, herhalingen=10, aantal=128)[-1].item()


class Gemengd(nn.Module):
    """Elke eenheid kiest zelf een mengsel van de aangeboden functies."""

    def __init__(self, breedte, namen):
        super().__init__()
        self.namen = namen
        self.fs = [FUNCTIES[n] for n in namen]
        self.logits = nn.Parameter(torch.zeros(len(namen), breedte))

    def forward(self, x):
        w = torch.softmax(self.logits, 0)
        return sum(w[i] * f(x) for i, f in enumerate(self.fs))

    @torch.no_grad()
    def telling(self):
        return torch.bincount(torch.softmax(self.logits, 0).argmax(0),
                              minlength=len(self.namen))


class Vast(nn.Module):
    def __init__(self, naam):
        super().__init__(); self.f = FUNCTIES[naam]
    def forward(self, x):
        return self.f(x)


def zet_activatie(model, maak):
    for blok in model.lagen:
        blok.feedforward.net[1] = maak()
    return model


if __name__ == "__main__":
    tok, train_ids, test_ids = laad_data()
    print(f"{len(train_ids)/1e6:.1f}M train / {len(test_ids)/1e6:.1f}M test karakters, "
          f"vocab {tok.vocab_size}")
    print(f"n_embed={N_EMBED} n_lagen={N_LAGEN} ff-breedte={FF_BREEDTE} "
          f"budget={BUDGET} stappen={N_STAPPEN}\n")

    print("=" * 64)
    print("1. GEBRUIK-OF-VERLIES op de feedforward")
    print("=" * 64)
    print(f"  {'aanpak':>30}{'loss':>9}{'params':>11}")
    for naam, verversen in (("dynamisch groeien/afsterven", True),
                            ("vast willekeurig deel", False)):
        verliezen = []
        for seed in SEEDS:
            m = bouw(tok.vocab_size, seed)
            maskers = [torch.zeros(FF_BREEDTE, dtype=torch.bool, device=APPARAAT)
                       for _ in range(N_LAGEN)]
            keuzes = [torch.randperm(FF_BREEDTE, device=APPARAAT)[:BUDGET]
                      for _ in range(N_LAGEN)]
            maskers = zet_aan(m, maskers, keuzes, schaal=0.5)
            t0 = time.time()
            m = train(m, train_ids, maskers, verversen, seed=seed)
            verliezen.append(eind_loss(m, test_ids, tok))
        actief = sum(p.numel() for p in m.parameters()) - \
                 N_LAGEN * (FF_BREEDTE - BUDGET) * (N_EMBED * 2 + 1)
        print(f"  {naam:>30}{sum(verliezen)/len(verliezen):>9.4f}{actief:>11,d}"
              f"   ({time.time()-t0:.0f}s/run)")

    verliezen = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = bouw(tok.vocab_size, seed)
        for blok in m.lagen:
            blok.feedforward.net[0] = nn.Linear(N_EMBED, BUDGET).to(APPARAAT)
            blok.feedforward.net[2] = nn.Linear(BUDGET, N_EMBED).to(APPARAAT)
        m = train(m, train_ids, seed=seed)
        verliezen.append(eind_loss(m, test_ids, tok))
    print(f"  {'dicht model op budget':>30}{sum(verliezen)/len(verliezen):>9.4f}"
          f"{sum(p.numel() for p in m.parameters()):>11,d}")

    m = train(bouw(tok.vocab_size, 0), train_ids, seed=0)
    print(f"  {'volle breedte (ijkpunt)':>30}{eind_loss(m, test_ids, tok):>9.4f}"
          f"{sum(p.numel() for p in m.parameters()):>11,d}")

    print("\n" + "=" * 64)
    print("2. WELKE ACTIVATIEFUNCTIE KIEST TAAL?")
    print("=" * 64)
    print(f"  {'activatie':>30}{'loss':>9}")
    for naam in FUNCTIES:
        m = zet_activatie(bouw(tok.vocab_size, 0), lambda: Vast(naam))
        m = train(m, train_ids, seed=0)
        print(f"  {naam:>30}{eind_loss(m, test_ids, tok):>9.4f}")

    namen = list(FUNCTIES)
    m = bouw(tok.vocab_size, 0)
    mengsels = [Gemengd(FF_BREEDTE, namen).to(APPARAAT) for _ in range(N_LAGEN)]
    for blok, g in zip(m.lagen, mengsels):
        blok.feedforward.net[1] = g
    m = train(m, train_ids, seed=0)
    print(f"  {'vrije keuze per eenheid':>30}{eind_loss(m, test_ids, tok):>9.4f}")
    print(f"\n  wat kiest elke laag? (eenheden van {FF_BREEDTE} per functie)")
    print(f"  {'laag':>6}" + "".join(f"{n:>11}" for n in namen))
    for i, g in enumerate(mengsels):
        print(f"  {i:>6}" + "".join(f"{int(v):>11}" for v in g.telling()))
