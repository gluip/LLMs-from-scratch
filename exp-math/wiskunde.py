# Kan een transformer optellen leren, of leert hij de tabel uit z'n hoofd?
#
# data/simple.txt bevat de volledige optel-tabel: 100 regels "a + b = c" voor
# a, b van 0 t/m 9. We houden 20 regels achter. Als het model die goed krijgt
# zonder ze ooit gezien te hebben, heeft het iets over de structuur van
# optellen geleerd in plaats van 80 regels onthouden.
#
# Uitgeklede kopie van ../experiment/exp.py: geen RoPE, geen dropout, geen
# genereren, geen plotten. Dit spoor mag zijn eigen kant op zonder dat het
# taalmodel daar meeverandert.

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config: alle instelbare knoppen op één plek
# ---------------------------------------------------------------------------
APPARAAT = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_BESTAND = Path(__file__).parent / "data" / "simple.txt"

LENGTE = 4                  # het venster: "9 + 9 =" is 4 tokens, het antwoord is de 5e
N_TEST = 20                 # regels die we achterhouden, willekeurig getrokken
SPLITS_SEED = 42            # vaste seed, zodat dezelfde 20 regels achterblijven

LOSS_SOORT = "kwadratisch_verschil"  # of "kruisentropie", zie bereken_loss()

N_EMBED = 32                # 64 haalde in de sweep iets vaker '9 + 9 = 18' goed
                            # (98% vs 95% gemiddeld), maar dat is precies de ene som
                            # die buiten het getrainde bereik valt — zie experiment 2.
                            # De ondergrens is gelijk, dus 32 blijft de default.
N_LAGEN = 1                 # één attention-laag om mee te beginnen
N_KOPPEN = 4
FF_FACTOR = 4.0
GEBRUIK_POSITIE = True      # 4 posities, elk een geleerde vector
GEBRUIK_FEEDFORWARD = True
GEBRUIK_LAYERNORM = False   # UIT, en dat is de grootste vondst van dit spoor. Met
                            # layernorm haalde het model 95% en legde het de cijfers op
                            # een BOOG; zonder wordt het een rechte, gelijkmatig verdeelde
                            # lijn en gaan alle 10 seeds naar 100% - inclusief 9+9=18, dat
                            # met layernorm altijd misging. Reden: layernorm normaliseert
                            # de lengte van elke vector weg, dus kan alleen de RICHTING
                            # nog informatie dragen, en richtingen liggen op een cirkel.
                            # Let op: dit geldt bij n_lagen=1. Layernorm zit er juist voor
                            # diepe stacks, dus zet hem terug aan als je lagen toevoegt.
                            # Zie EXPERIMENTEN.md, experiment 4.

BATCH_AANTAL = 16           # regels per stap, willekeurig uit de trainingsset.
                            # None = full-batch (alle 80). Full-batch is niet
                            # vertekend — elke stap ziet alles — maar mist de
                            # gradient-ruis van mini-batches, en juist die ruis
                            # remt memoriseren wat af. Bij een experiment dat
                            # memoriseren-versus-generaliseren meet wil je dat
                            # als knop, niet als stilzwijgende keuze.
LEERRATE = 3e-3
WEIGHT_DECAY = 0.3          # de belangrijkste knop van allemaal. Op AdamW's default
                            # (0,01) haalde het model gemiddeld 84% op de
                            # achtergehouden sommen, maar met een spreiding van 40%
                            # tot 100% over niets dan de startgewichten. Op 0,3 is
                            # die spreiding weg: elke seed haalt 19 of 20 van de 20.
                            # Zie EXPERIMENTEN.md, experiment 2.
N_STAPPEN = 10000           # 3000 was te kort om het effect van weight decay te laten
                            # inzakken: bij wd=0,3 ging de ondergrens van 85% naar 95%
EVAL_INTERVAL = 250
SEEDS = range(10)           # met 20 testvragen springt de accuratesse in stappen
                            # van 5%; één run zegt dus niets. Dit model is
                            # minuscuul, dus 10 runs kosten niets.


class SomTokenizer:
    """Tokenizer op getalniveau: elk heel getal is één token, geen cijfer.

    De spatie is hier de scheiding *tussen* tokens: we splitsen erop en
    gooien hem daarna weg, hij wordt zelf geen token. Daardoor is "18" één
    token en niet twee, en is elke regel precies 5 tokens lang:

        "9 + 9 = 18"  ->  ['9', '+', '9', '=', '18']

    Zo is er geen enkel afgekapt antwoord, en past het hele probleem in één
    voorspelling.

    De vocabulaire (21 tokens) komt uit de data zelf: '0'..'9' als operand,
    '10'..'18' die alleen als antwoord voorkomen, plus '+' en '='. Let op wat
    dat betekent voor het model: '12' is voor het netwerk een willekeurig
    symbool. Niets vertelt het dat het "twaalf" is, of dat het tussen '11' en
    '13' in hoort. Dat moet het afleiden uit welke sommen erop uitkomen —
    precies daar zit de vraag of het generaliseert.
    """

    def __init__(self, regels):
        tokens = {t for regel in regels for t in regel}
        # sorteren op getalwaarde waar dat kan, zodat de vocabulaire leesbaar
        # is bij het printen. Het model merkt van de volgorde niets: het zijn
        # en blijven losse ids.
        self.tokens = sorted(tokens, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t))
        self.token_naar_int = {t: i for i, t in enumerate(self.tokens)}
        self.int_naar_token = {i: t for t, i in self.token_naar_int.items()}

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, regel):
        return [self.token_naar_int[t] for t in regel]

    def decode(self, ids):
        return [self.int_naar_token[i] for i in ids]

    def waarden(self):
        """Getalwaarde per token-id, of nan voor '+' en '='.

        Nodig voor de kwadratisch-verschil-loss: die rekent met het getal dat
        een token voorstelt, niet met het id. Token-id 13 hoeft niet het getal
        13 te zijn.
        """
        return torch.tensor(
            [float(t) if t.isdigit() else float("nan") for t in self.tokens]
        )


def laad_sommen(bestand=DATA_BESTAND):
    """Leest de sommen en geeft ze terug als lijst van token-lijsten."""
    regels = [r.split() for r in bestand.read_text(encoding="utf-8").splitlines() if r.strip()]
    assert regels, f"{bestand} is leeg"
    assert all(len(r) == LENGTE + 1 for r in regels), \
        f"elke regel moet {LENGTE + 1} tokens zijn, gevonden: {sorted({len(r) for r in regels})}"
    return regels


def splits_train_test(x, y, n_test=N_TEST, seed=SPLITS_SEED):
    """Willekeurige splitsing over de regels, met vaste seed."""
    g = torch.Generator().manual_seed(seed)
    volgorde = torch.randperm(len(x), generator=g)
    test_idx, train_idx = volgorde[:n_test], volgorde[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def maak_batch(x, y, aantal=BATCH_AANTAL, generator=None):
    """Trek `aantal` willekeurige regels. `aantal=None` geeft alles terug.

    Anders dan in exp.py trekken we hier geen vensters uit één doorlopende
    tokenstroom: onze data bestaat uit losse regels die niets met elkaar te
    maken hebben.
    """
    if aantal is None or aantal >= len(x):
        return x, y
    idx = torch.randint(0, len(x), (aantal,), generator=generator, device=x.device)
    return x[idx], y[idx]


class AffiniteitsLaag(nn.Module):
    """Eén attention-laag: (batch, T, n_embed) erin, dezelfde vorm eruit.

    Volledig Q/K/V: "waar zoek ik naar" (Q), "wat bied ik aan" (K) en "wat
    neem ik mee" (V) hebben elk hun eigen geleerde matrix. In exp.py zijn dat
    nog aan/uit-vlaggen omdat daar het opbouwen zelf het experiment was; hier
    staan ze vast aan.

    `n_embed` wordt opgesplitst in `n_koppen` koppen die elk hun eigen
    affiniteit berekenen over hun eigen stuk van de vector; `uit_proj` (W_o)
    mengt ze daarna weer.

    Het causale masker blijft aan: positie i mag alleen naar i en eerder
    kijken. Voor dit probleem maakt het weinig uit — we gebruiken toch alleen
    de laatste positie, en die mag naar alles kijken — maar het houdt de laag
    gelijk aan die in exp.py.
    """

    def __init__(self, n_embed, n_koppen):
        super().__init__()
        assert n_embed % n_koppen == 0, "n_embed moet deelbaar zijn door n_koppen"
        self.n_koppen = n_koppen
        self.kop_dim = n_embed // n_koppen
        self.Q = nn.Linear(n_embed, n_embed, bias=False)
        self.K = nn.Linear(n_embed, n_embed, bias=False)
        self.V = nn.Linear(n_embed, n_embed, bias=False)
        self.uit_proj = nn.Linear(n_embed, n_embed)

    def _splits(self, t):
        """(batch, T, n_embed) -> (batch, n_koppen, T, kop_dim)."""
        B, T, _ = t.shape
        return t.view(B, T, self.n_koppen, self.kop_dim).transpose(1, 2)

    def forward(self, h):
        B, T, C = h.shape
        q, k, v = self._splits(self.Q(h)), self._splits(self.K(h)), self._splits(self.V(h))
        affiniteit = q @ k.transpose(-2, -1) / self.kop_dim ** 0.5  # (batch, n_koppen, T, T)

        masker = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        affiniteit = affiniteit.masked_fill(masker, float("-inf"))  # niet vooruitkijken

        gewichten = torch.softmax(affiniteit, dim=-1)
        uit = gewichten @ v                                        # (batch, n_koppen, T, kop_dim)
        uit = uit.transpose(1, 2).contiguous().view(B, T, C)       # koppen weer aan elkaar
        return self.uit_proj(uit), gewichten[:, 0]                 # eerste kop, voor inspectie


class FeedForwardLaag(nn.Module):
    """Verbreedt naar FF_FACTOR x n_embed, niet-lineariteit, en weer terug.

    Werkt per positie apart — mengen tussen posities doet de attention-laag.
    Dit is de plek waar de eigenlijke som uitgerekend moet worden: attention
    haalt de twee operanden bij elkaar, dit zet ze om in een antwoord.
    """

    def __init__(self, n_embed, factor=FF_FACTOR):
        super().__init__()
        binnen = max(1, int(round(n_embed * factor)))
        self.net = nn.Sequential(
            nn.Linear(n_embed, binnen),
            nn.ReLU(),
            nn.Linear(binnen, n_embed),
        )

    def forward(self, h):
        return self.net(h)


class Blok(nn.Module):
    """Attention + feedforward, allebei als residu, met layernorm ervoor."""

    def __init__(self, n_embed, n_koppen, gebruik_feedforward=True, gebruik_layernorm=True):
        super().__init__()
        self.attentie = AffiniteitsLaag(n_embed, n_koppen)
        self.feedforward = FeedForwardLaag(n_embed) if gebruik_feedforward else None
        self.ln1 = nn.LayerNorm(n_embed) if gebruik_layernorm else None
        self.ln2 = nn.LayerNorm(n_embed) if (gebruik_layernorm and gebruik_feedforward) else None

    def forward(self, h):
        attentie_uit, gewichten = self.attentie(self.ln1(h) if self.ln1 is not None else h)
        h = h + attentie_uit  # residu, op het ongenormaliseerde pad
        if self.feedforward is not None:
            h = h + self.feedforward(self.ln2(h) if self.ln2 is not None else h)
        return h, gewichten


class SomModel(nn.Module):
    """Embedding -> n_lagen blokken -> één voorspelling voor het antwoord.

    De uitvoerkop hangt af van `loss_soort`:

      "kwadratisch_verschil"  -> nn.Linear(n_embed, 1): één getal, bv. 17.4
      "kruisentropie"         -> nn.Linear(n_embed, vocab_size): een score per token

    Alleen de laatste positie doet ertoe. Zie het plaatje bij bereken_loss().
    """

    def __init__(self, vocab_size, loss_soort=LOSS_SOORT, n_embed=N_EMBED, n_lagen=N_LAGEN,
                 n_koppen=N_KOPPEN, gebruik_positie=GEBRUIK_POSITIE,
                 gebruik_feedforward=GEBRUIK_FEEDFORWARD, gebruik_layernorm=GEBRUIK_LAYERNORM,
                 lengte=LENGTE):
        super().__init__()
        self.gebruik_positie = gebruik_positie
        self.embed = nn.Embedding(vocab_size, n_embed)
        if gebruik_positie:
            self.pos_embed = nn.Embedding(lengte, n_embed)
        self.lagen = nn.ModuleList([
            Blok(n_embed, n_koppen, gebruik_feedforward, gebruik_layernorm) for _ in range(n_lagen)
        ])
        self.uit = nn.Linear(n_embed, 1 if loss_soort == "kwadratisch_verschil" else vocab_size)

    def forward(self, x):
        h = self.embed(x)  # (batch, T, n_embed)
        if self.gebruik_positie:
            h = h + self.pos_embed(torch.arange(x.shape[1], device=x.device))
        for laag in self.lagen:
            h, gewichten = laag(h)
        return self.uit(h[:, -1]), gewichten  # alleen de laatste positie


def bereken_loss(uitvoer, y, y_waarde, loss_soort=LOSS_SOORT):
    """Loss op het antwoord.

        regel:      9      +      9      =      18
        index:      0      1      2      3       4
                    \\___ invoer x (4 tokens) __/   \\_ doel y _/

    Het antwoord is het 5e token (index 4), maar de voorspelling ervan komt
    uit de 4e invoerpositie (index 3, het '='): een transformer voorspelt
    vanaf elke positie het token dat erná komt. Vandaar h[:, -1] in forward().

    De tussenliggende posities meetrainen ('9' -> '+') doen we niet: dat is
    ruis en deels niet eens voorspelbaar.

    "kwadratisch_verschil" is het kwadraat van (voorspelling - antwoord) —
    zie toon_waarom_kwadraat() voor waarom er een kwadraat omheen moet, en
    waarom dat iets anders is dan het verschil van de kwadraten.
    """
    if loss_soort == "kwadratisch_verschil":
        return F.mse_loss(uitvoer.squeeze(-1), y_waarde)
    return F.cross_entropy(uitvoer, y)


def voorspelde_waarde(uitvoer, token_waarden, loss_soort=LOSS_SOORT):
    """Het antwoord dat het model geeft, als getal.

    Bij "kwadratisch_verschil" geeft het model een getal als 17,4; afronden
    maakt er een antwoord van. Bij "kruisentropie" kiest het een token en
    zoeken we op welk getal dat is.
    """
    if loss_soort == "kwadratisch_verschil":
        return uitvoer.squeeze(-1).round()
    return token_waarden[uitvoer.argmax(dim=-1)]


@torch.no_grad()
def accuratesse(model, x, y_waarde, token_waarden, loss_soort=LOSS_SOORT):
    """Aandeel sommen dat exact goed is. Loss is hier de verkeerde hoofdmaat:
    je wilt weten of de som klopt, niet hoe zeker het model was."""
    model.eval()
    uitvoer, _ = model(x)
    goed = (voorspelde_waarde(uitvoer, token_waarden, loss_soort) == y_waarde).float().mean().item()
    model.train()
    return goed


def toon_waarom_kwadraat():
    """Laat zien waarom een kale 'voorspelling - antwoord' niet werkt.

    Het naïeve idee is: de loss is gewoon het verschil tussen wat het model
    zegt en wat het antwoord is. Reken dat uit over een batch en je ziet
    meteen wat er misgaat.
    """
    voorspelling = torch.tensor([10.0, 4.0])
    antwoord = torch.tensor([7.0, 7.0])
    verschil = voorspelling - antwoord

    print("\nwaarom niet gewoon 'voorspelling - antwoord'?")
    print("  som 3 + 4 = 7    model zegt 10    ->  10 - 7 = +3")
    print("  som 2 + 5 = 7    model zegt  4    ->   4 - 7 = -3")
    print(f"\n  gemiddeld verschil     : {verschil.mean():>6.1f}   <- 'foutloos', maar allebei 3 mis")
    print(f"  gemiddeld |verschil|   : {verschil.abs().mean():>6.1f}")
    print(f"  gemiddeld verschil^2   : {verschil.pow(2).mean():>6.1f}   <- deze nemen we")
    print("\n  De twee afwijkingen heffen elkaar op: de ene zit omhoog, de andere")
    print("  omlaag. Loss 0 geeft geen gradient, dus het model leert hier niets van.")
    print("  Een kwadraat kan niet wegvallen, en laat grote fouten zwaarder wegen.")

    print("\n  kwadraat van het verschil != verschil van de kwadraten:")
    print(f"    {'voorspelling':>13}{'antwoord':>10}{'(v-a)^2':>10}{'v^2-a^2':>10}")
    for v, a in [(10, 7), (4, 7), (0, 7)]:
        print(f"    {v:>13}{a:>10}{(v - a) ** 2:>10}{v ** 2 - a ** 2:>10}")
    print("  (v-a)^2 kan nooit negatief worden en is 0 alleen als het antwoord klopt.")
    print("  v^2-a^2 kan dat wel, en is het laagst bij v=0 wat het antwoord ook is:")
    print("  daar duwt gradient descent elke uitvoer naar 0 en leert het model niets.")


def train(x_train, y_train, y_train_waarde, x_test, y_test, y_test_waarde,
          tokenizer, token_waarden, seed=0, loss_soort=LOSS_SOORT, stil=False,
          weight_decay=WEIGHT_DECAY, n_stappen=N_STAPPEN, **model_knoppen):
    """Traint één model en geeft het terug, met de accuratesse-geschiedenis.

    `model_knoppen` gaat door naar SomModel (n_lagen, n_embed, ...), zodat
    vergelijkingsscripts kunnen sweepen zonder de config-constanten te patchen.
    """
    torch.manual_seed(seed)
    model = SomModel(tokenizer.vocab_size, loss_soort=loss_soort, **model_knoppen).to(x_train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEERRATE, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_stappen)
    g = torch.Generator(device=x_train.device).manual_seed(seed)

    geschiedenis = []
    for stap in range(n_stappen):
        idx = torch.randint(0, len(x_train), (BATCH_AANTAL,), generator=g, device=x_train.device) \
            if BATCH_AANTAL is not None and BATCH_AANTAL < len(x_train) else slice(None)
        uitvoer, _ = model(x_train[idx])
        loss = bereken_loss(uitvoer, y_train[idx], y_train_waarde[idx], loss_soort)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if stap % EVAL_INTERVAL == 0 or stap == n_stappen - 1:
            acc_train = accuratesse(model, x_train, y_train_waarde, token_waarden, loss_soort)
            acc_test = accuratesse(model, x_test, y_test_waarde, token_waarden, loss_soort)
            geschiedenis.append((stap, acc_train, acc_test))
            if not stil:
                print(f"  stap {stap:>5}  loss {loss.item():>8.4f}  "
                      f"train {acc_train:>5.0%}  test {acc_test:>5.0%}")
    return model, geschiedenis


@torch.no_grad()
def toon_fouten(model, x_test, y_test_waarde, tokenizer, token_waarden, loss_soort=LOSS_SOORT):
    """Print alle achtergehouden sommen met het gegeven antwoord.

    Hoe hij ernaast zit is informatiever dan dat hij ernaast zit: '7+5=13' is
    een ander soort fout dan '7+5=3'.
    """
    model.eval()
    uitvoer, _ = model(x_test)
    gegeven = voorspelde_waarde(uitvoer, token_waarden, loss_soort)
    ruw = uitvoer.squeeze(-1) if loss_soort == "kwadratisch_verschil" else None
    model.train()

    print("\n  de 20 achtergehouden sommen:")
    for i in range(len(x_test)):
        som = " ".join(tokenizer.decode(x_test[i].tolist()))
        klopt = gegeven[i] == y_test_waarde[i]
        extra = f"  (ruw {ruw[i]:>6.2f})" if ruw is not None else ""
        print(f"    {som:>10} {gegeven[i]:>4.0f}   "
              f"{'goed' if klopt else f'FOUT, moest {y_test_waarde[i]:.0f}':<18}{extra}")


if __name__ == "__main__":
    regels = laad_sommen()
    tokenizer = SomTokenizer(regels)

    print(f"regels:      {len(regels)}")
    print(f"vocab size:  {tokenizer.vocab_size}")
    print(f"vocabulaire: {tokenizer.tokens}")

    # round-trip: encode -> decode moet elke regel teruggeven
    for r in regels:
        assert tokenizer.decode(tokenizer.encode(r)) == r
    assert tokenizer.vocab_size == 21, tokenizer.vocab_size
    print("round-trip check ok")

    ids = torch.tensor([tokenizer.encode(r) for r in regels], dtype=torch.long)
    x, y = ids[:, :LENGTE], ids[:, LENGTE]

    print(f"\nvoorbeeld: {' '.join(regels[99])}")
    print(f"  invoer x (4 tokens): {tokenizer.decode(x[99].tolist())}")
    print(f"  doel   y (5e token): {tokenizer.decode([y[99].item()])[0]}")

    token_waarden = tokenizer.waarden()
    x_train, y_train, x_test, y_test = splits_train_test(x, y)
    assert len(x_train) + len(x_test) == len(x) == 100
    # geen overlap: elke regel zit in train of in test, nooit in allebei
    alles = {tuple(r.tolist()) for r in torch.cat([x_train, x_test])}
    assert len(alles) == 100, "train en test overlappen"
    print(f"\nsplitsing (seed {SPLITS_SEED}): {len(x_train)} train, {len(x_test)} test")

    toon_waarom_kwadraat()

    x_train, y_train = x_train.to(APPARAAT), y_train.to(APPARAAT)
    x_test, y_test = x_test.to(APPARAAT), y_test.to(APPARAAT)
    token_waarden = token_waarden.to(APPARAAT)
    y_train_waarde, y_test_waarde = token_waarden[y_train], token_waarden[y_test]

    n_par = sum(p.numel() for p in SomModel(tokenizer.vocab_size).parameters())
    print(f"\nmodel: n_lagen={N_LAGEN} n_embed={N_EMBED} n_koppen={N_KOPPEN} "
          f"({n_par} parameters), loss={LOSS_SOORT}, {N_STAPPEN} stappen op {APPARAAT}")

    print(f"\nseed {list(SEEDS)[0]} in detail:")
    model, _ = train(x_train, y_train, y_train_waarde, x_test, y_test, y_test_waarde,
                     tokenizer, token_waarden, seed=list(SEEDS)[0])
    toon_fouten(model, x_test, y_test_waarde, tokenizer, token_waarden)

    # met 20 testvragen springt de accuratesse in stappen van 5%: één run is ruis
    print(f"\nover {len(list(SEEDS))} seeds:")
    resultaten = []
    for seed in SEEDS:
        _, geschiedenis = train(x_train, y_train, y_train_waarde, x_test, y_test, y_test_waarde,
                                tokenizer, token_waarden, seed=seed, stil=True)
        _, acc_train, acc_test = geschiedenis[-1]
        resultaten.append((acc_train, acc_test))
        print(f"  seed {seed}:  train {acc_train:>5.0%}  test {acc_test:>5.0%}")

    trains = torch.tensor([t for t, _ in resultaten])
    tests = torch.tensor([t for _, t in resultaten])
    print(f"\n  train: {trains.mean():.1%} (min {trains.min():.0%}, max {trains.max():.0%})")
    print(f"  test : {tests.mean():.1%} (min {tests.min():.0%}, max {tests.max():.0%})")
