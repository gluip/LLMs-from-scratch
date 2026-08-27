# Optellen, aftrekken, vermenigvuldigen - met per bewerking het kleinst
# mogelijke model dat het foutloos doet.
#
# Aanleiding: voor optellen bleek vrijwel alles overbodig wat een transformer
# onderscheidend maakt (zie EXPERIMENTEN.md, experiment 5 en 6). Dat kwam
# doordat optellen twee bijzondere eigenschappen heeft:
#
#   commutatief : a + b = b + a   ->  het model hoeft a en b niet uit elkaar
#                                     te houden, dus positie-informatie kan weg
#   lineair     : de uitkomst is een gewogen som van de invoer  ->  er is geen
#                                     niet-lineariteit nodig, dus de feedforward
#                                     kan weg
#
# Aftrekken mist de eerste eigenschap, vermenigvuldigen de tweede. Dit script
# toetst of het model daardoor precies terugkrijgt wat het bij optellen kwijt
# kon. Draaien:  .venv/bin/python -u exp-math/rekenen.py

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).parent / "data"
BEWERKINGEN = {                      # naam -> (bestand, teken)
    "optellen":         ("simple.txt", "+"),
    "aftrekken":        ("aftrekken.txt", "-"),
    "vermenigvuldigen": ("vermenigvuldigen.txt", "*"),
    "beide":            ("optellen_aftrekken.txt", "+/-"),   # 200 regels, twee bewerkingen door elkaar
    "drie":             ("drie_bewerkingen.txt", "+/-/*"),    # 300 regels, alle drie door elkaar
}

LENGTE = 4                  # "9 * 9 =" is 4 tokens, het antwoord is de 5e
TEST_FRACTIE = 0.2          # aandeel achtergehouden regels. Bij 100 regels is dat
                            # 20 (zoals altijd); bij de gecombineerde set van 200
                            # regels 40, zodat de verhouding gelijk blijft.
SPLITS_SEED = 42
LEERRATE = 3e-3
WEIGHT_DECAY = 0.3
N_STAPPEN = 10000
BATCH_AANTAL = 16
SEEDS = range(5)


def is_getal(t):
    """Ook '-9' is een getal; str.isdigit() zegt daar nee op."""
    return t.lstrip("-").isdigit()


class Tokenizer:
    """Elk heel getal is één token, ook een negatief getal.

    De spatie is de scheiding tússen tokens: we splitsen erop en gooien hem
    weg. Daardoor is '-9' één token en niet twee, net zoals '18' dat was bij
    optellen.
    """

    def __init__(self, regels):
        tokens = {t for regel in regels for t in regel}
        self.tokens = sorted(tokens, key=lambda t: (not is_getal(t),
                                                    int(t) if is_getal(t) else t))
        self.naar_int = {t: i for i, t in enumerate(self.tokens)}

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, regel):
        return [self.naar_int[t] for t in regel]

    def decode(self, ids):
        return [self.tokens[i] for i in ids]

    def waarden(self):
        """Getalwaarde per token-id; nan voor de tekens '+', '-', '*' en '='."""
        return torch.tensor([float(t) if is_getal(t) else float("nan")
                             for t in self.tokens])


def laad(bewerking):
    """Leest een bewerking in en splitst hem in train en test.

    Geeft terug: tokenizer, token-waarden, en x/y voor train en test. De y's
    zijn meteen getalwaarden (geen token-ids), want de loss rekent met het
    getal dat een token voorstelt.
    """
    bestand, _ = BEWERKINGEN[bewerking]
    regels = [r.split() for r in (DATA / bestand).read_text(encoding="utf-8").splitlines() if r.strip()]
    assert all(len(r) == LENGTE + 1 for r in regels), f"{bestand}: niet elke regel is 5 tokens"
    tok = Tokenizer(regels)
    ids = torch.tensor([tok.encode(r) for r in regels], dtype=torch.long)
    x, y = ids[:, :LENGTE], ids[:, LENGTE]
    g = torch.Generator().manual_seed(SPLITS_SEED)
    volgorde = torch.randperm(len(x), generator=g)
    n_test = round(TEST_FRACTIE * len(x))
    test, train = volgorde[:n_test], volgorde[n_test:]
    tw = tok.waarden()
    return tok, tw, x[train], tw[y[train]], x[test], tw[y[test]]


class Rekenmodel(nn.Module):
    """Het uitgeklede model, met de drie onderdelen die optellen niet nodig
    had als losse knoppen.

    `positie`: geeft elke plek in de reeks een eigen geleerde vector. Zonder
    dit zijn "a op plek 0" en "a op plek 2" niet te onderscheiden — prima voor
    optellen, fataal voor aftrekken.

    `leer_aandacht`: met Q en K bepaalt het model zelf waar het naar kijkt.
    Zonder deze knop wordt de aandacht vervangen door een vaste middeling over
    alle posities. Bij optellen bleek de geleerde aandacht uit zichzelf al
    uniform te worden, dus daar scheelde het niets.

    `ff`: een feedforward-laag met ReLU, de enige niet-lineariteit in het
    model. Zonder dit is het hele netwerk een lineaire afbeelding, en dan kan
    het per definitie geen a*b uitrekenen.

    `n_koppen` en `uit_proj`: de aandacht opsplitsen in meerdere koppen die
    elk hun eigen stuk van de vector vullen, en die daarna weer mengen met
    W_o. Dit tweetal is wat aftrekken mogelijk maakt: softmax-gewichten zijn
    altijd niet-negatief, dus met één kop is er geen manier om ergens een min
    voor te zetten. Met twee koppen kan de ene `a` in de eerste helft van de
    vector zetten en de andere `b` in de tweede, en kan W_o daar tegengestelde
    tekens aan geven. Dat is precies waar multi-head attention voor bedoeld is.
    """

    def __init__(self, vocab_size, n_embed, positie=False, leer_aandacht=False, ff=False,
                 n_koppen=1, uit_proj=False, getekend=False, soort="softmax"):
        super().__init__()
        # `getekend=True` is de oude naam voor soort="getekend"; blijft werken
        # zodat experiment 8 letterlijk reproduceerbaar blijft.
        self.soort = "getekend" if getekend else soort
        assert self.soort in ("softmax", "getekend", "gecentreerd", "verschil", "tanh")
        assert n_embed % n_koppen == 0, "n_embed moet deelbaar zijn door n_koppen"
        self.n_embed, self.n_koppen = n_embed, n_koppen
        self.kop_dim = n_embed // n_koppen
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos = nn.Embedding(LENGTE, n_embed) if positie else None
        if leer_aandacht:
            self.Q = nn.Linear(n_embed, n_embed, bias=False)
            self.K = nn.Linear(n_embed, n_embed, bias=False)
        else:
            self.Q = self.K = None
        # soort "verschil" trekt twee losse softmaxen van elkaar af en heeft
        # daarvoor een tweede stel Q/K nodig
        if leer_aandacht and self.soort == "verschil":
            self.Q2 = nn.Linear(n_embed, n_embed, bias=False)
            self.K2 = nn.Linear(n_embed, n_embed, bias=False)
        else:
            self.Q2 = self.K2 = None
        self.V = nn.Linear(n_embed, n_embed, bias=False)
        self.W_o = nn.Linear(n_embed, n_embed) if uit_proj else None
        self.ff = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
                                nn.Linear(4 * n_embed, n_embed)) if ff else None
        self.uit = nn.Linear(n_embed, 1)

    def _splits(self, t):
        """(batch, T, n_embed) -> (batch, n_koppen, T, kop_dim)."""
        B, T, _ = t.shape
        return t.view(B, T, self.n_koppen, self.kop_dim).transpose(1, 2)

    def _gewichten(self, aff, h):
        """Van affiniteiten naar gewichten, in vijf smaken.

        Softmax levert altijd niet-negatieve gewichten die optellen tot 1: de
        uitvoer is dan een gewogen GEMIDDELDE van de values, en ligt dus binnen
        hun omhullende. Een verschil v(a) - v(b) ligt daarbuiten, en daarom kan
        gewone softmax-aandacht niet aftrekken (zie EXPERIMENTEN.md, exp. 7-8).

        De vier alternatieven laten wel minnen toe, maar betalen daar elk iets
        anders voor. In de tabel is T de lengte van de reeks:

          getekend      2*g - 1            som = 2 - T, groeit dus met de lengte
          gecentreerd   g - gemiddelde(g)  som = 0, maar een teken kan niet
                                           sterker worden dan 1/T
          verschil      softmax1 - softmax2  som = 0, en elke helft mag volledig
                                           concentreren; kost een extra Q/K
          tanh          tanh(aff)          geen normalisatie; het model moet de
                                           schaal zelf in de hand houden
        """
        if self.soort == "tanh":
            return torch.tanh(aff)
        if self.soort == "verschil":
            q2, k2 = self._splits(self.Q2(h)), self._splits(self.K2(h))
            aff2 = q2 @ k2.transpose(-2, -1) / math.sqrt(self.kop_dim)
            return torch.softmax(aff, -1) - torch.softmax(aff2, -1)
        g = torch.softmax(aff, -1)
        if self.soort == "getekend":
            return 2 * g - 1
        if self.soort == "gecentreerd":
            return g - g.mean(-1, keepdim=True)
        return g

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(T, device=x.device))
        v = self.V(h)
        if self.Q is not None:
            q, k, vs = self._splits(self.Q(h)), self._splits(self.K(h)), self._splits(v)
            aff = q @ k.transpose(-2, -1) / math.sqrt(self.kop_dim)
            g = self._gewichten(aff, h)
            meng = (g @ vs).transpose(1, 2).contiguous().view(B, T, -1)
        else:
            meng = v.mean(1, keepdim=True).expand(B, T, -1)   # vaste middeling
        if self.W_o is not None:
            meng = self.W_o(meng)
        h = h + meng
        if self.ff is not None:
            h = h + self.ff(h)
        return self.uit(h)[:, -1].squeeze(-1)


def train(xtr, ytr, xte, yte, vocab_size, seed=0, n_stappen=N_STAPPEN, **knoppen):
    """Traint één model en geeft het terug met train- en test-accuratesse."""
    torch.manual_seed(seed)
    model = Rekenmodel(vocab_size, **knoppen)
    opt = torch.optim.AdamW(model.parameters(), lr=LEERRATE, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_stappen)
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_stappen):
        i = torch.randint(0, len(xtr), (BATCH_AANTAL,), generator=g)
        loss = F.mse_loss(model(xtr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        acc_tr = (model(xtr).round() == ytr).float().mean().item()
        acc_te = (model(xte).round() == yte).float().mean().item()
    return model, acc_tr, acc_te


def meet(bewerking, seeds=SEEDS, n_stappen=N_STAPPEN, **knoppen):
    """Train- en test-accuratesse over meerdere seeds, plus het aantal parameters.

    `knoppen` gaat naar het model, `n_stappen` naar de training — vandaar dat
    die hier apart staat en niet in de **knoppen zit.
    """
    tok, tw, xtr, ytr, xte, yte = laad(bewerking)
    tr, te = [], []
    for seed in seeds:
        _, a_tr, a_te = train(xtr, ytr, xte, yte, tok.vocab_size, seed=seed,
                              n_stappen=n_stappen, **knoppen)
        tr.append(a_tr); te.append(a_te)
    n_par = sum(p.numel() for p in Rekenmodel(tok.vocab_size, **knoppen).parameters())
    return torch.tensor(tr), torch.tensor(te), n_par


def naam_van(knoppen):
    """Korte omschrijving van welke knoppen aanstaan."""
    aan = [k.replace("leer_aandacht", "aandacht") for k in ("positie", "leer_aandacht", "ff")
           if knoppen.get(k)]
    if knoppen.get("uit_proj"):
        aan.append("W_o")
    if knoppen.get("n_koppen", 1) > 1:
        aan.append(f"{knoppen['n_koppen']} koppen")
    soort = "getekend" if knoppen.get("getekend") else knoppen.get("soort", "softmax")
    if soort != "softmax":
        aan.append(soort)
    return "kaal" if not aan else " + ".join(aan)


# ---------------------------------------------------------------------------
# De zoektocht: welke onderdelen heeft elke bewerking echt nodig?
# ---------------------------------------------------------------------------
LADDER = [
    {},
    {"positie": True},
    {"leer_aandacht": True},
    {"ff": True},
    {"positie": True, "leer_aandacht": True},
    {"positie": True, "ff": True},
    {"leer_aandacht": True, "ff": True},
    {"positie": True, "leer_aandacht": True, "ff": True},
]


def zoek(bewerking, n_embed=16):
    """Fase 1: welke knoppen zijn nodig? Alles op een ruime n_embed."""
    print(f"\n{'=' * 72}\n{bewerking.upper()}   (n_embed={n_embed})\n{'=' * 72}")
    print(f"{'onderdelen':>34}{'train':>8}{'test':>8}{'min':>7}{'params':>9}")
    uitslag = []
    for knoppen in LADDER:
        tr, te, n = meet(bewerking, n_embed=n_embed, **knoppen)
        uitslag.append((naam_van(knoppen), knoppen, te.mean().item(), te.min().item(), n))
        print(f"{naam_van(knoppen):>34}{tr.mean():>8.0%}{te.mean():>8.0%}{te.min():>7.0%}{n:>9,d}")
    return uitslag


def krimp(bewerking, knoppen, maten=(16, 8, 4, 2)):
    """Fase 2: met de nodige knoppen aan, hoe klein kan n_embed?"""
    print(f"\n  krimpen met '{naam_van(knoppen)}':")
    print(f"  {'n_embed':>9}{'test':>8}{'min':>7}{'params':>9}")
    bodem = None
    for n_embed in maten:
        tr, te, n = meet(bewerking, n_embed=n_embed, **knoppen)
        vlag = ""
        if te.min() == 1.0:
            bodem = (n_embed, n)
            vlag = "  <- foutloos"
        print(f"  {n_embed:>9}{te.mean():>8.0%}{te.min():>7.0%}{n:>9,d}{vlag}")
    return bodem


if __name__ == "__main__":
    samenvatting = {}
    for bewerking in BEWERKINGEN:
        uitslag = zoek(bewerking)
        # de goedkoopste combinatie die bij elke seed foutloos is; anders de beste
        perfect = [(naam, kn, n) for naam, kn, gem, mn, n in uitslag if mn == 1.0]
        if perfect:
            naam, knoppen, _ = min(perfect, key=lambda r: r[2])
            print(f"\n  goedkoopste foutloze combinatie: '{naam}'")
            bodem = krimp(bewerking, knoppen)
        else:
            naam, knoppen, gem, mn, n = max(uitslag, key=lambda r: (r[2], -r[4]))
            print(f"\n  GEEN foutloze combinatie; beste is '{naam}' met {gem:.0%}")
            bodem = None
        samenvatting[bewerking] = (naam, bodem)

    print(f"\n{'=' * 72}\nSAMENVATTING\n{'=' * 72}")
    print(f"{'bewerking':>18}{'nodige onderdelen':>36}{'n_embed':>9}{'params':>9}")
    for bewerking, (naam, bodem) in samenvatting.items():
        ne, np_ = bodem if bodem else ("-", "-")
        print(f"{bewerking:>18}{naam:>36}{str(ne):>9}{str(np_):>9}")
