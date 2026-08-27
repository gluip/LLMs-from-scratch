# Waarom springt de test-accuratesse van 40% naar 100% op alleen de seed?
#
# Experiment 1 (zie EXPERIMENTEN.md) liet zien dat het model met de
# kwadratisch-verschil-loss gemiddeld 84% haalt, maar met een spreiding van
# 40% tot 100% over niets anders dan de startgewichten. Dit script zoekt de
# knop die de ONDERGRENS omhoog haalt — niet het gemiddelde. Een model dat
# gemiddeld goed is maar bij een op de vijf seeds instort, is niet bruikbaar.
#
# Draaien:  .venv/bin/python exp-math/stabiliteit_vgl.py

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import wiskunde as w

SWEEP_SEEDS = range(5)    # genoeg om de spreiding te zien
FINALE_SEEDS = range(10)  # de winnende config nog eens, steviger gemeten


def meet(xtr, ytr, ytrw, xte, yte, ytew, tok, tw, seeds, **knoppen):
    """Test-accuratesse per seed, plus welke sommen misgingen.

    Beide in één doorloop: elk model is duur genoeg (10000 stappen) om het
    niet twee keer te willen trainen, één keer voor het cijfer en één keer
    voor de fouten.
    """
    accs, fouten = [], {}
    for seed in seeds:
        model, geschiedenis = w.train(xtr, ytr, ytrw, xte, yte, ytew, tok, tw,
                                      seed=seed, stil=True, **knoppen)
        accs.append(geschiedenis[-1][2])
        model.eval()
        with torch.no_grad():
            gegeven = w.voorspelde_waarde(model(xte)[0], tw)
        for i in range(len(xte)):
            if gegeven[i] != ytew[i]:
                fouten.setdefault(" ".join(tok.decode(xte[i].tolist())), []).append(int(gegeven[i]))
    return torch.tensor(accs), fouten


def regel(label, a):
    print(f"  {label:<34} {a.mean():>7.0%}{a.min():>7.0%}{a.max():>7.0%}")


if __name__ == "__main__":
    regels = w.laad_sommen()
    tok = w.SomTokenizer(regels)
    ids = torch.tensor([tok.encode(r) for r in regels], dtype=torch.long)
    x, y = ids[:, :w.LENGTE], ids[:, w.LENGTE]
    xtr, ytr, xte, yte = w.splits_train_test(x, y)
    tw = tok.waarden()
    args = (xtr, ytr, tw[ytr], xte, yte, tw[yte], tok, tw)
    t0 = time.time()

    # -----------------------------------------------------------------------
    # 1. Weight decay: trekt gewichten naar 0 tenzij de data ze nodig heeft.
    #    Memoriseren kost veel losse, grote gewichten; een regel die voor alle
    #    sommen werkt kost er minder. Straf op grootte maakt de regel dus
    #    aantrekkelijker dan de tabel.
    # -----------------------------------------------------------------------
    print(f"\n{'weight decay x trainingsduur':<36}{'gem':>7}{'min':>7}{'max':>7}")
    for stappen in (3000, 10000):
        for wd in (0.0, 0.01, 0.1, 0.3, 1.0):
            a, _ = meet(*args, seeds=SWEEP_SEEDS, weight_decay=wd, n_stappen=stappen)
            regel(f"wd={wd:<5} {stappen} stappen", a)

    # -----------------------------------------------------------------------
    # 2. Helpt meer capaciteit nog, bovenop de beste weight decay?
    # -----------------------------------------------------------------------
    print(f"\n{'model, bij wd=0.3 en 10000 stappen':<36}{'gem':>7}{'min':>7}{'max':>7}")
    for n_lagen, n_embed in ((1, 16), (1, 32), (1, 64), (2, 32)):
        a, _ = meet(*args, seeds=SWEEP_SEEDS, weight_decay=0.3, n_stappen=10000,
                    n_lagen=n_lagen, n_embed=n_embed)
        regel(f"n_lagen={n_lagen} n_embed={n_embed}", a)

    # -----------------------------------------------------------------------
    # 3. De winnende config over 10 seeds, met de fouten erbij.
    # -----------------------------------------------------------------------
    print(f"\n{'finale: wd=0.3, 10000 stappen':<36}{'gem':>7}{'min':>7}{'max':>7}")
    a, fout_teller = meet(*args, seeds=FINALE_SEEDS, weight_decay=0.3, n_stappen=10000)
    regel(f"over {len(list(FINALE_SEEDS))} seeds", a)

    print("\n  welke sommen gaan mis:")
    for som, gaven in sorted(fout_teller.items(), key=lambda kv: -len(kv[1])):
        print(f"    {som:>10}  {len(gaven)}/{len(list(FINALE_SEEDS))} seeds fout, gaven {gaven}")

    # Het plafond zit niet in het model maar in de splitsing: 9+9 is de enige
    # som die 18 oplevert, en die zit in de testset. Het hoogste antwoord dat
    # het model ooit heeft moeten produceren is dus 17. Bij een regressie-kop
    # is 18 daarmee extrapolatie voorbij het bekende bereik.
    print(f"\n  antwoorden in train: {tw[ytr].min():.0f} t/m {tw[ytr].max():.0f}   "
          f"in test: {tw[yte].min():.0f} t/m {tw[yte].max():.0f}")
    print(f"  komt 18 voor in train? {bool((tw[ytr] == 18).any())}  "
          f"(alleen 9+9 geeft 18, en die is achtergehouden)")
    print(f"\n{time.time() - t0:.0f}s")
