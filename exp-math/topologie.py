# Kan het netwerk zijn eigen omvang leren? Snoeien en laten aangroeien.
#
# Idee: eenheden die weinig doen weghalen (pruning), en eenheden die het druk
# hebben splitsen (growing / Net2Net). Beide bestaan in de literatuur; dit
# script bouwt de eenvoudigste eerlijke versie voor de feedforward-laag van
# het reken-model.
#
# De cruciale controle is NIET "wordt het model kleiner zonder verlies" - dat
# lukt bijna altijd - maar "haalt snoeien iets wat je met een even klein model
# vanaf nul niet ook had gekregen". Zonder die vergelijking meet je alleen dat
# je te groot begonnen was.
#
# Draaien:  .venv/bin/python -u exp-math/topologie.py   (~20 min)

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import rekenen as r

BASIS = dict(n_embed=32, positie=True, leer_aandacht=True, uit_proj=True,
             n_koppen=8, ff=True)
STAPPEN = 30000
BIJSCHAVEN = 4000        # stappen na elke snoeironde


def ff_breedte(model):
    return model.ff[0].out_features


@torch.no_grad()
def belang(model, x):
    """Hoe belangrijk is elke verborgen eenheid van de feedforward?

    Twee dingen samen: hoeveel varieert de eenheid over de invoer (een eenheid
    die altijd hetzelfde doet draagt geen informatie), en hoe zwaar telt hij
    mee in de uitgaande gewichten. Een eenheid die hard schommelt maar nergens
    naartoe gaat is net zo nutteloos als een die stilstaat.
    """
    h = model.embed(x) + model.pos(torch.arange(x.shape[1]))
    q, k, vs = model._splits(model.Q(h)), model._splits(model.K(h)), model._splits(model.V(h))
    g = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(model.kop_dim), -1)
    meng = (g @ vs).transpose(1, 2).contiguous().view(len(x), x.shape[1], -1)
    h1 = h + model.W_o(meng)
    verborgen = torch.relu(model.ff[0](h1))[:, -1]
    return verborgen.std(0) * model.ff[2].weight.abs().mean(0)


@torch.no_grad()
def snoei(model, masker, hoeveel, x):
    """Zet de `hoeveel` minst belangrijke nog-levende eenheden uit."""
    score = belang(model, x).clone()
    score[~masker] = float("inf")            # al gesnoeide eenheden overslaan
    weg = score.argsort()[:hoeveel]
    masker[weg] = False
    pas_masker_toe(model, masker)
    return masker


@torch.no_grad()
def pas_masker_toe(model, masker):
    """Gesnoeide eenheden echt op nul, in- én uitgaand."""
    model.ff[0].weight[~masker] = 0
    model.ff[0].bias[~masker] = 0
    model.ff[2].weight[:, ~masker] = 0


@torch.no_grad()
def groei(model, masker, hoeveel, x):
    """Splits de drukste eenheden: kopieer ze naar een lege plek.

    Net2Net: de nieuwe eenheid krijgt dezelfde ingaande gewichten (met een
    tikje ruis, anders blijven ze voor altijd identiek) en beide krijgen de
    halve uitgaande gewichten. Het model doet op dat moment dus precies
    hetzelfde als ervoor - de splitsing is gratis, en pas daarna kunnen de
    twee helften uit elkaar groeien.
    """
    leeg = (~masker).nonzero().squeeze(1)
    if len(leeg) == 0:
        return masker
    score = belang(model, x).clone()
    score[~masker] = -float("inf")
    druk = score.argsort(descending=True)[:min(hoeveel, len(leeg))]
    for bron, doel in zip(druk, leeg):
        model.ff[0].weight[doel] = model.ff[0].weight[bron] * (1 + 0.01 * torch.randn_like(model.ff[0].weight[bron]))
        model.ff[0].bias[doel] = model.ff[0].bias[bron]
        model.ff[2].weight[:, bron] /= 2
        model.ff[2].weight[:, doel] = model.ff[2].weight[:, bron]
        masker[doel] = True
    return masker


def train_met_masker(model, xtr, ytr, masker, stappen, seed=0, lr=r.LEERRATE):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=r.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stappen)
    g = torch.Generator().manual_seed(seed)
    for _ in range(stappen):
        i = torch.randint(0, len(xtr), (r.BATCH_AANTAL,), generator=g)
        loss = F.mse_loss(model(xtr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if masker is not None:
            pas_masker_toe(model, masker)     # gesnoeid blijft gesnoeid
    return model


@torch.no_grad()
def meet(model, x, y):
    model.eval()
    a = (model(x).round() == y).float().mean().item()
    model.train()
    return a


def parameters_bij(model, masker):
    """Parameters die echt meedoen: gesnoeide rijen en kolommen tellen niet."""
    totaal = sum(p.numel() for p in model.parameters())
    dood = int((~masker).sum())
    per_eenheid = model.ff[0].in_features + 1 + model.ff[2].out_features
    return totaal - dood * per_eenheid


if __name__ == "__main__":
    tok, tw, xtr, ytr, xte, yte = r.laad("drie")
    print("SNOEIEN — feedforward van 128 eenheden, stapsgewijs terug\n")

    torch.manual_seed(0)
    model = r.Rekenmodel(tok.vocab_size, **BASIS)
    model = train_met_masker(model, xtr, ytr, None, STAPPEN)
    masker = torch.ones(ff_breedte(model), dtype=torch.bool)
    print(f"  {'levend':>8}{'test':>8}{'params':>9}   (na volledig trainen)")
    print(f"  {int(masker.sum()):>8}{meet(model, xte, yte):>8.0%}"
          f"{parameters_bij(model, masker):>9,d}")

    verloop = []
    for doel in (96, 64, 48, 32, 24, 16, 8):
        weg = int(masker.sum()) - doel
        masker = snoei(model, masker, weg, xtr)
        na_snoei = meet(model, xte, yte)
        model = train_met_masker(model, xtr, ytr, masker, BIJSCHAVEN, seed=doel)
        na_schaven = meet(model, xte, yte)
        verloop.append((doel, na_snoei, na_schaven, parameters_bij(model, masker)))
        print(f"  {doel:>8}{na_schaven:>8.0%}{parameters_bij(model, masker):>9,d}"
              f"   (direct na snoeien {na_snoei:.0%})")

    print("\nDE CONTROLE — even groot, maar vanaf nul getraind\n")
    print(f"  {'ff-breedte':>11}{'gesnoeid':>10}{'vanaf nul':>11}")
    vanaf_nul = {}
    for doel, _, na_schaven, _ in verloop:
        if doel not in (64, 32, 16, 8):
            continue
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            m2 = r.Rekenmodel(tok.vocab_size, **BASIS)
            # zelfde architectuur, maar de ff-laag meteen op de kleine maat
            m2.ff = torch.nn.Sequential(torch.nn.Linear(32, doel), torch.nn.ReLU(),
                                        torch.nn.Linear(doel, 32))
            m2 = train_met_masker(m2, xtr, ytr, None, STAPPEN, seed=seed)
            accs.append(meet(m2, xte, yte))
        vanaf_nul[doel] = sum(accs) / len(accs)
        print(f"  {doel:>11}{na_schaven:>10.0%}{vanaf_nul[doel]:>11.0%}")

    print("\nGROEIEN — de drukste eenheden splitsen (Net2Net)\n")
    torch.manual_seed(0)
    m3 = r.Rekenmodel(tok.vocab_size, **BASIS)
    m3.ff = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 32))
    klein = torch.zeros(128, dtype=torch.bool); klein[:16] = True
    pas_masker_toe(m3, klein)
    m3 = train_met_masker(m3, xtr, ytr, klein, STAPPEN // 2)
    print(f"  {'levend':>8}{'test':>8}   startpunt: 16 eenheden")
    print(f"  {int(klein.sum()):>8}{meet(m3, xte, yte):>8.0%}")
    for ronde in range(3):
        voor = meet(m3, xte, yte)
        klein = groei(m3, klein, int(klein.sum()), xtr)
        direct = meet(m3, xte, yte)
        m3 = train_met_masker(m3, xtr, ytr, klein, BIJSCHAVEN * 2, seed=100 + ronde)
        print(f"  {int(klein.sum()):>8}{meet(m3, xte, yte):>8.0%}"
              f"   (vlak na splitsen {direct:.0%}, was {voor:.0%})")
