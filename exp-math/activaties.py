# Mag elke node zijn eigen activatiefunctie kiezen?
#
# De feedforward-laag is het enige onderdeel dat kan buigen, en welke bocht
# hij kan maken hangt af van de activatiefunctie. Voor vermenigvuldigen is er
# een voor de hand liggende kandidaat, want een product is uit kwadraten te
# bouwen:
#
#     a·b = ( (a+b)² - (a-b)² ) / 4
#
# Een ReLU moet die kromme met rechte stukjes benaderen; x² levert hem in één
# keer. Dit script toetst dat, en bouwt daarna een laag waarin elke eenheid
# zelf een mengsel van functies mag kiezen - dan kun je achteraf uitlezen
# waar het netwerk voor koos.
#
# Draaien:  .venv/bin/python -u exp-math/activaties.py   (~15 min)

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import rekenen as r

BASIS = dict(n_embed=32, positie=True, leer_aandacht=True, uit_proj=True,
             n_koppen=8, ff=True)
BREEDTE = 128
STAPPEN = 30000
SEEDS = range(3)

# x² groeit hard; delen door 2 houdt de schaal in de buurt van de andere
# functies, zodat de vergelijking over de activatie gaat en niet over toevallig
# betere conditionering
FUNCTIES = {
    "relu":     torch.relu,
    "kwadraat": lambda x: x * x / 2,
    "tanh":     torch.tanh,
    "gelu":     F.gelu,
    "lineair":  lambda x: x,
}


class GemengdeActivatie(nn.Module):
    """Elke eenheid kiest zelf een mengsel van de aangeboden functies.

    Per eenheid staan er evenveel logits als er functies zijn; een softmax
    daarover geeft de mengverhouding. Bij het begin is alles gelijk verdeeld,
    en tijdens het trainen kan elke eenheid naar zijn eigen voorkeur toe
    schuiven. Achteraf is die voorkeur gewoon uit te lezen.
    """

    def __init__(self, breedte, namen):
        super().__init__()
        self.namen = namen
        self.functies = [FUNCTIES[n] for n in namen]
        self.logits = nn.Parameter(torch.zeros(len(namen), breedte))

    def forward(self, x):
        w = torch.softmax(self.logits, 0)
        return sum(w[i] * f(x) for i, f in enumerate(self.functies))

    @torch.no_grad()
    def voorkeur(self):
        """Gemiddelde mengverhouding, en hoeveel eenheden welke functie kiezen."""
        w = torch.softmax(self.logits, 0)
        keuze = w.argmax(0)
        return w.mean(1), torch.bincount(keuze, minlength=len(self.namen))


def maak(tok, activatie, breedte=BREEDTE, seed=0):
    torch.manual_seed(seed)
    m = r.Rekenmodel(tok.vocab_size, **BASIS)
    m.ff = nn.Sequential(nn.Linear(BASIS["n_embed"], breedte), activatie,
                         nn.Linear(breedte, BASIS["n_embed"]))
    return m


def train(m, xtr, ytr, seed=0, stappen=STAPPEN):
    opt = torch.optim.AdamW(m.parameters(), lr=r.LEERRATE, weight_decay=r.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stappen)
    g = torch.Generator().manual_seed(seed)
    for _ in range(stappen):
        i = torch.randint(0, len(xtr), (r.BATCH_AANTAL,), generator=g)
        loss = F.mse_loss(m(xtr[i]), ytr[i])
        if not torch.isfinite(loss):
            return m, False              # x² kan ontsporen; dat willen we zien
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m, True


@torch.no_grad()
def meet(m, x, y):
    m.eval(); a = (m(x).round() == y).float().mean().item(); m.train(); return a


class Vast(nn.Module):
    def __init__(self, naam):
        super().__init__(); self.f = FUNCTIES[naam]
    def forward(self, x):
        return self.f(x)


if __name__ == "__main__":
    tok, tw, xtr, ytr, xte, yte = r.laad("drie")

    print("VASTE ACTIVATIE — welke bocht past bij vermenigvuldigen?\n")
    print(f"  {'functie':>10}{'test':>8}{'min':>7}{'gelukt':>9}")
    for naam in FUNCTIES:
        accs, ok = [], 0
        for seed in SEEDS:
            m, gelukt = train(maak(tok, Vast(naam), seed=seed), xtr, ytr, seed=seed)
            if gelukt:
                accs.append(meet(m, xte, yte)); ok += 1
        if accs:
            a = torch.tensor(accs)
            print(f"  {naam:>10}{a.mean():>8.0%}{a.min():>7.0%}{ok}/{len(list(SEEDS)):>8}")
        else:
            print(f"  {naam:>10}{'—':>8}{'—':>7}{ok}/{len(list(SEEDS)):>8}  (liep vast)")

    print("\n\nVRIJE KEUZE — elke eenheid kiest zelf een mengsel\n")
    namen = ["relu", "kwadraat", "tanh", "lineair"]
    for seed in SEEDS:
        act = GemengdeActivatie(BREEDTE, namen)
        m = maak(tok, act, seed=seed)
        m, gelukt = train(m, xtr, ytr, seed=seed)
        if not gelukt:
            print(f"  seed {seed}: liep vast"); continue
        gemiddeld, telling = act.voorkeur()
        print(f"  seed {seed}: test {meet(m, xte, yte):.0%}")
        print(f"    {'functie':>10}{'gem. gewicht':>14}{'eenheden die hem kiezen':>26}")
        for i, n in enumerate(namen):
            print(f"    {n:>10}{gemiddeld[i]:>14.3f}{int(telling[i]):>19} van {BREEDTE}")
