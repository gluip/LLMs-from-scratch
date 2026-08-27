# Willekeurig laten aangroeien, en wat niet gebruikt wordt sterft af.
#
# Experiment 13 liet zien dat snoeien verliest van vanaf-nul trainen, en dat
# groeien via kopieren niets oplevert. Beide hadden hetzelfde probleem: de
# vorm werd van buitenaf bepaald, met een aparte meting per stap.
#
# Dit is een andere aanpak. Houd een vast budget aan actieve eenheden. Gooi
# elke zoveel stappen de zwakste eruit en zet er evenveel WILLEKEURIGE nieuwe
# voor in de plaats. Je hoeft een nieuwe eenheid niet te evalueren: als hij
# niets bijdraagt is hij de volgende ronde weer de zwakste en verdwijnt hij
# vanzelf. De training zelf is de selectie.
#
# Dat is ook waarom het goedkoop is. Blind muteren is duur zodra je elke
# kandidaat apart moet bijschaven om hem te beoordelen (honderden trainingen);
# hier loopt de selectie mee in de ene training die je toch al deed.
#
# Draaien:  .venv/bin/python -u exp-math/gebruik_of_verlies.py   (~20 min)

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import rekenen as r
import topologie as t

BASIS = t.BASIS
BREEDTE = 128            # hoeveel plekken er zijn
BUDGET = 32              # hoeveel er tegelijk actief mogen zijn
STAPPEN = 30000
INTERVAL = 1000          # om de hoeveel stappen verversen
VERVERS = 0.20           # welk deel van het budget per ronde vervangen wordt
SEEDS = range(3)


def nieuw_model(tok, seed=0):
    torch.manual_seed(seed)
    m = r.Rekenmodel(tok.vocab_size, **BASIS)
    m.ff = nn.Sequential(nn.Linear(BASIS["n_embed"], BREEDTE), nn.ReLU(),
                         nn.Linear(BREEDTE, BASIS["n_embed"]))
    return m


@torch.no_grad()
def zet_aan(model, masker, plekken, schaal=0.05):
    """Wek plekken op met kleine willekeurige gewichten.

    Uitgaande gewichten mogen niet exact nul zijn: de gradient op de ingaande
    gewichten is evenredig met de uitgaande, dus een eenheid die met nul
    begint krijgt nooit een duw en blijft voor altijd liggen.
    """
    for p in plekken:
        model.ff[0].weight[p] = torch.randn_like(model.ff[0].weight[p]) * schaal
        model.ff[0].bias[p] = 0
        model.ff[2].weight[:, p] = torch.randn_like(model.ff[2].weight[:, p]) * schaal
        masker[p] = True
    return masker


@torch.no_grad()
def ververs(model, masker, x, aantal):
    """De zwakste `aantal` actieve eenheden vervangen door willekeurige nieuwe."""
    score = t.belang(model, x).clone()
    score[~masker] = float("inf")
    weg = score.argsort()[:aantal]
    masker[weg] = False
    t.pas_masker_toe(model, masker)
    leeg = (~masker).nonzero().squeeze(1)
    nieuw = leeg[torch.randperm(len(leeg))[:aantal]]
    return zet_aan(model, masker, nieuw), int((~masker).sum())


def train_dynamisch(m, xtr, ytr, xte, yte, seed=0, verversen=True):
    """Normale training, met om de INTERVAL stappen een verversingsronde."""
    masker = torch.zeros(BREEDTE, dtype=torch.bool)
    start = torch.randperm(BREEDTE)[:BUDGET]
    masker = zet_aan(m, masker, start, schaal=0.5)
    opt = torch.optim.AdamW(m.parameters(), lr=r.LEERRATE, weight_decay=r.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAPPEN)
    g = torch.Generator().manual_seed(seed)
    n_vervangen = 0
    for stap in range(STAPPEN):
        i = torch.randint(0, len(xtr), (r.BATCH_AANTAL,), generator=g)
        verlies = F.mse_loss(m(xtr[i]), ytr[i])
        opt.zero_grad(); verlies.backward(); opt.step(); sch.step()
        t.pas_masker_toe(m, masker)
        # in de laatste vijfde niet meer verversen: dan mag het bezinken
        if verversen and stap and stap % INTERVAL == 0 and stap < STAPPEN * 0.8:
            masker, _ = ververs(m, masker, xtr, max(1, int(BUDGET * VERVERS)))
            n_vervangen += max(1, int(BUDGET * VERVERS))
    return m, masker, n_vervangen


if __name__ == "__main__":
    tok, tw, xtr, ytr, xte, yte = r.laad("drie")
    print(f"GEBRUIK-OF-VERLIES — {BUDGET} actieve eenheden van {BREEDTE} plekken")
    print(f"elke {INTERVAL} stappen wordt {VERVERS:.0%} vervangen door willekeurige nieuwe\n")

    uitslag = {}

    accs = []
    for seed in SEEDS:
        m, masker, n = train_dynamisch(nieuw_model(tok, seed), xtr, ytr, xte, yte, seed=seed)
        accs.append(t.meet(m, xte, yte))
    uitslag["dynamisch groeien/afsterven"] = (torch.tensor(accs), n)
    print(f"  dynamisch: {torch.tensor(accs).mean():.0%} "
          f"(min {torch.tensor(accs).min():.0%}), {n} vervangingen onderweg")

    accs = []
    for seed in SEEDS:
        m, masker, _ = train_dynamisch(nieuw_model(tok, seed), xtr, ytr, xte, yte,
                                       seed=seed, verversen=False)
        accs.append(t.meet(m, xte, yte))
    uitslag["vast willekeurig deel"] = (torch.tensor(accs), 0)
    print(f"  vast willekeurig deel van 32: {torch.tensor(accs).mean():.0%} "
          f"(min {torch.tensor(accs).min():.0%})")

    accs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = r.Rekenmodel(tok.vocab_size, **BASIS)
        m.ff = nn.Sequential(nn.Linear(32, BUDGET), nn.ReLU(), nn.Linear(BUDGET, 32))
        m = t.train_met_masker(m, xtr, ytr, None, STAPPEN, seed=seed)
        accs.append(t.meet(m, xte, yte))
    uitslag["dicht, vanaf nul"] = (torch.tensor(accs), 0)
    print(f"  dicht model van 32, vanaf nul: {torch.tensor(accs).mean():.0%} "
          f"(min {torch.tensor(accs).min():.0%})")

    print("\n  ter vergelijking uit experiment 13:")
    print("    stapsgewijs snoeien naar 32          87%")
    print("    in één keer snoeien naar 32          95%")

    print(f"\n{'=' * 58}")
    print(f"{'aanpak':>34}{'test':>8}{'min':>8}")
    for naam, (a, _) in uitslag.items():
        print(f"{naam:>34}{a.mean():>8.0%}{a.min():>8.0%}")
