# dit is een experiment in een ander soort woordvoorspeller
# we trainen op de the verdict en proberen aldoor het
# maak een tokeniner op basis van the-verdict.txt

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config: alle instelbare knoppen op één plek
# ---------------------------------------------------------------------------
TEKST_PAD = Path(__file__).parent / "pinkeltje_schoon.txt"
TRAIN_FRACTIE = 0.9        # aandeel van de tekst dat train wordt, de rest is test

LENGTE = 20                 # context length; 5, 40 en 60 gaven allemaal dezelfde loss — 20 is genoeg
BATCH_AANTAL = 64           # stukjes per trainings-batch
TEST_BATCH_AANTAL = 256     # stukjes per test-batch (groter = stabielere meting)

N_EMBED = 20                 # dimensies per karakter-embedding
N_LAGEN = 5                    # 10 gaf maar 0,025 nats winst voor 2x de rekentijd — niet de moeite waard
GEBRUIK_POSITIE = True        # positie-embedding: weet het model hoe ver terug iets stond?
GEBRUIK_FEEDFORWARD = True    # feedforward-bottleneck-laag na elke attention-laag
LOSSE_QK = True                # aparte Q- en K-matrix i.p.v. één gedeelde W
LOSSE_V = True                 # aparte V-matrix i.p.v. V = Q
GEBRUIK_LAYERNORM = True        # normaliseer h vlak voor elke sublaag
GEBRUIK_MASKER = True           # causaal masker: nooit vooruitkijken (altijd aan houden)
DROPOUT = 0.0                   # geen overfitting-probleem op deze dataset om tegen te vechten

LEERRATE = 1e-2               # startpunt van de cosine-decay
N_STAPPEN = 3000
EVAL_INTERVAL = 300
SEED = 0

N_EMBED_OPTIES = [20, 40]  # de knop die we in dit experiment vergelijken


class CharTokenizer:
    """Tokenizer op karakterniveau: elk karakter wordt een integer."""

    def __init__(self, tekst):
        # de vocabulaire is de gesorteerde set unieke karakters
        self.chars = sorted(set(tekst))
        self.char_naar_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_naar_char = {i: ch for ch, i in self.char_naar_int.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, tekst):
        return [self.char_naar_int[ch] for ch in tekst]

    def decode(self, ids):
        return "".join(self.int_naar_char[i] for i in ids)


def maak_batch(ids, lengte=20, aantal=10, generator=None):
    """Trek `aantal` willekeurige stukjes van `lengte` tokens, met hun targets.

    Geeft twee tensors van vorm (aantal, lengte) terug:
      x = de karakters zelf
      y = dezelfde karakters, één positie opgeschoven (het "volgende" karakter)
    Zo hoort bij elke positie i in x meteen het juiste antwoord y[i].
    """
    # we hebben lengte+1 tokens nodig per stukje: lengte input + 1 extra target
    starts = torch.randint(0, len(ids) - lengte, (aantal,), generator=generator)
    x = torch.stack([ids[s:s + lengte] for s in starts])
    y = torch.stack([ids[s + 1:s + lengte + 1] for s in starts])
    return x, y


class AffiniteitsLaag(nn.Module):
    """Eén stackbare attention-laag: neemt (batch, T, n_embed) en geeft ook
    (batch, T, n_embed) terug, zodat je er meerdere achter elkaar kunt zetten.

    Met `losse_qk=False` (default) gebruiken we net als voorheen maar één
    gedeelde matrix W voor zowel "wie zoek ik" als "wie bied ik aan": elke
    vector wordt ermee geprojecteerd, en de affiniteit tussen positie i en j
    is het inproduct van die ene projectie met zichzelf. Nadeel daarvan
    (zoals we eerder zagen): het inproduct van een vector met zichzelf is
    vrijwel altijd het grootst, dus elke positie let vooral op zichzelf.
    Met `losse_qk=True` krijgen Q en K allebei hun eigen matrix, zodat "waar
    ik naar zoek" en "wat ik aanbied" losgekoppeld zijn.

    Met `losse_v=False` (default) is de V (value) nog gelijk aan diezelfde
    projectie die ook voor de affiniteit gebruikt wordt. Met `losse_v=True`
    krijgt V zijn eigen geleerde matrix — dan is "waar let ik op" (affiniteit)
    ook losgekoppeld van "wat neem ik mee" (value).

    losse_qk=True + losse_v=True samen is het volledige Q/K/V zoals bij echte
    attention; allebei False is de "alles-in-1"-versie waarmee we begonnen.

    Met `gebruik_masker=True` (default) mag positie i alleen naar i en eerder
    kijken. Zonder masker kan positie i ook de toekomst zien — en dat is
    binnen één trainings-venster vrijwel altijd een leugentje om bestwil: het
    doel y[i] staat voor bijna elke i letterlijk op positie i+1 van diezelfde
    invoer, dus zonder masker kan het model "kopieer gewoon de volgende
    letter" leren in plaats van echt voorspellen. Bij genereren bestaat die
    volgende letter nog niet (die moet je juist produceren), dus die
    truc werkt daar niet — vandaar dat masking niet optioneel is bij echte
    autoregressieve modellen.

    `dropout` (0 tot 1) zet na de softmax willekeurig een deel van de
    aandacht-gewichten op 0, alleen tijdens training. Dat dwingt het model om
    niet blind op één specifieke eerdere letter te leunen, maar het patroon
    over meerdere mogelijke letters te leren herkennen — minder kans om de
    trainingsdata letterlijk te onthouden.
    """

    def __init__(self, n_embed, losse_qk=False, losse_v=False, gebruik_masker=True, dropout=0.0):
        super().__init__()
        self.n_embed = n_embed
        self.losse_qk = losse_qk
        self.gebruik_masker = gebruik_masker
        self.dropout = nn.Dropout(dropout)
        if losse_qk:
            self.Q = nn.Linear(n_embed, n_embed, bias=False)
            self.K = nn.Linear(n_embed, n_embed, bias=False)
        else:
            self.W = nn.Linear(n_embed, n_embed, bias=False)  # de ene gedeelde matrix
        self.V = nn.Linear(n_embed, n_embed, bias=False) if losse_v else None

    def forward(self, h):
        # h: (batch, T, n_embed) -> ook weer (batch, T, n_embed), dus stackbaar
        T = h.shape[1]
        if self.losse_qk:
            q = self.Q(h)
            k = self.K(h)
            affiniteit = q @ k.transpose(-2, -1)  # (batch, T, T), inproduct query x key
            basis = q  # terugvaloptie voor v, als losse_v uitstaat
        else:
            q = k = self.W(h)               # (batch, T, n_embed), geprojecteerd
            affiniteit = q @ k.transpose(-2, -1)           # (batch, T, T), inproduct per paar
            basis = q
        affiniteit = affiniteit / self.n_embed ** 0.5   # schaling, zoals bij echte attention

        if self.gebruik_masker:
            masker = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            affiniteit = affiniteit.masked_fill(masker, float("-inf"))  # niet vooruitkijken

        gewichten = torch.softmax(affiniteit, dim=-1)  # per rij: kansen die optellen tot 1
        v = self.V(h) if self.V is not None else basis  # losse matrix op h, of terugval
        output = self.dropout(gewichten) @ v  # (batch, T, n_embed); dropout alleen voor de output-berekening
        return output, gewichten  # ongewijzigde gewichten teruggeven, voor inspectie


class FeedForwardLaag(nn.Module):
    """Verlaagt de dimensie (delen door 4), niet-lineariteit, en weer terug naar
    n_embed. Ook stackbaar: (batch, T, n_embed) -> (batch, T, n_embed). Werkt
    per positie apart (geen menging tussen posities, dat doet de attention-laag)."""

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed // 4),
            nn.ReLU(),
            nn.Linear(n_embed // 4, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, h):
        return self.net(h)


class Blok(nn.Module):
    """Eén stackbaar blok: een AffiniteitsLaag, optioneel gevolgd door een
    FeedForwardLaag. (batch, T, n_embed) -> (batch, T, n_embed).

    Beide sub-lagen zijn residu-verbindingen: h = h + sublaag(h), niet
    h = sublaag(h). Zo kan een nog-niet-getrainde (of destructieve, zoals de
    smalle feedforward-bottleneck) sublaag zich gedragen als bijna-niks-doen,
    in plaats van verplicht alles te vervangen wat er al in h zat.

    Met `gebruik_layernorm=True` wordt h vlak vóór elke sublaag genormaliseerd
    (gemiddelde 0, spreiding 1, per positie over de n_embed-as) — de residu-
    optelling zelf gebeurt daarna op het ongenormaliseerde pad. Zonder dit kan
    de magnitude van h bij elke laag verder oplopen (h = h + iets, telkens
    opnieuw), wat bij een diepe stack (grote n_lagen) tot een instabiele start
    leidt — zoals we terugzagen als een hoge loss bij stap 0.
    """

    def __init__(self, n_embed, gebruik_feedforward=False, losse_qk=False, losse_v=False, gebruik_layernorm=False, gebruik_masker=True, dropout=0.0):
        super().__init__()
        self.attentie = AffiniteitsLaag(n_embed, losse_qk=losse_qk, losse_v=losse_v, gebruik_masker=gebruik_masker, dropout=dropout)
        self.feedforward = FeedForwardLaag(n_embed, dropout=dropout) if gebruik_feedforward else None
        self.ln1 = nn.LayerNorm(n_embed) if gebruik_layernorm else None
        self.ln2 = nn.LayerNorm(n_embed) if (gebruik_layernorm and gebruik_feedforward) else None

    def forward(self, h):
        h_voor_attentie = self.ln1(h) if self.ln1 is not None else h
        attentie_uit, gewichten = self.attentie(h_voor_attentie)
        h = h + attentie_uit  # residu, op het ongenormaliseerde pad
        if self.feedforward is not None:
            h_voor_ff = self.ln2(h) if self.ln2 is not None else h
            h = h + self.feedforward(h_voor_ff)  # ook residu
        return h, gewichten


class AffiniteitsModel(nn.Module):
    """Embedding -> n_lagen stackbare AffiniteitsLaag'en -> voorspelling per letter.

    Zonder positie-informatie weet de affiniteit alleen "lijkt dit karakter op
    dat karakter", niet "hoe ver terug stond het". Met `gebruik_positie=True`
    krijgt elke positie 0..T-1 zijn eigen geleerde vector (net als de letters
    dat hebben), opgeteld bij de letter-embedding — zodat het model afstand
    kan leren meewegen.
    """

    def __init__(self, vocab_size, n_embed=12, n_lagen=1, gebruik_positie=True, gebruik_feedforward=False, losse_qk=False, losse_v=False, gebruik_layernorm=False, gebruik_masker=True, dropout=0.0, max_lengte=128):
        super().__init__()
        self.gebruik_positie = gebruik_positie
        self.embed = nn.Embedding(vocab_size, n_embed)
        if gebruik_positie:
            self.pos_embed = nn.Embedding(max_lengte, n_embed)
        self.embed_dropout = nn.Dropout(dropout)
        self.lagen = nn.ModuleList([
            Blok(n_embed, gebruik_feedforward, losse_qk, losse_v, gebruik_layernorm, gebruik_masker, dropout) for _ in range(n_lagen)
        ])
        self.uit = nn.Linear(n_embed, vocab_size)  # enige plek die naar vocab_size gaat

    def forward(self, x):
        T = x.shape[1]
        h = self.embed(x)  # (batch, T, n_embed)
        if self.gebruik_positie:
            posities = torch.arange(T, device=x.device)
            h = h + self.pos_embed(posities)  # zelfde vector voor elke positie i, over de hele batch
        h = self.embed_dropout(h)
        for laag in self.lagen:
            h, gewichten = laag(h)  # zelfde vorm erin als eruit
        scores = self.uit(h)  # (batch, T, vocab_size), pas hier naar vocab_size
        return scores, gewichten  # gewichten van de laatste laag, voor inspectie


def genereer(model, tokenizer, start, n_nieuw=40, lengte=LENGTE, generator=None):
    """Genereer karakter voor karakter verder op `start`, door telkens uit de
    voorspelde kansverdeling te samplen (niet steeds de meest waarschijnlijke).

    De context wordt afgekapt op de laatste `lengte` tokens: dat is waar het
    model op getraind is. Zonder die afkapping groeit de invoer door tot voorbij
    `lengte`, en dan gebruikt het model rijen van pos_embed die tijdens training
    nooit aan bod kwamen — die staan dus nog op hun willekeurige startwaarde.
    """
    model.eval()
    ids = torch.tensor([tokenizer.encode(start)], dtype=torch.long)  # (1, T)
    with torch.no_grad():
        for _ in range(n_nieuw):
            scores, _ = model(ids[:, -lengte:])
            kansen = torch.softmax(scores[0, -1], dim=-1)  # kansen voor het laatste teken
            volgend = torch.multinomial(kansen, num_samples=1, generator=generator)
            ids = torch.cat([ids, volgend.unsqueeze(0)], dim=1)
    model.train()
    return tokenizer.decode(ids[0].tolist())


def train_affiniteitsmodel(
    n_lagen, n_embed, train_ids, test_ids, tokenizer, lengte=LENGTE,
    gebruik_positie=GEBRUIK_POSITIE, gebruik_feedforward=GEBRUIK_FEEDFORWARD,
    losse_qk=LOSSE_QK, losse_v=LOSSE_V, gebruik_layernorm=False, gebruik_masker=True, dropout=0.0,
    aantal_train=BATCH_AANTAL, aantal_test=TEST_BATCH_AANTAL,
    lr=LEERRATE, n_stappen=N_STAPPEN, eval_interval=EVAL_INTERVAL, seed=SEED,
):
    """Traint een AffiniteitsModel met `n_lagen` lagen, `n_embed` dimensies en context `lengte`, en houdt de loss bij.

    De leerrate volgt een cosine-schema: begint op `lr` en zakt geleidelijk
    naar ~0 aan het einde van de training. Vroeg in training mag je grote
    stappen zetten, maar tegen het einde (dicht bij een minimum) duwt een
    grote stap je er juist weer overheen — dat zagen we terug als ruis en
    niet-monotone loss bij een vaste hoge leerrate.
    """
    torch.manual_seed(seed)
    model = AffiniteitsModel(
        tokenizer.vocab_size, n_embed=n_embed, n_lagen=n_lagen,
        gebruik_positie=gebruik_positie, gebruik_feedforward=gebruik_feedforward, losse_qk=losse_qk, losse_v=losse_v,
        gebruik_layernorm=gebruik_layernorm, gebruik_masker=gebruik_masker, dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_stappen)

    genereer_start = tokenizer.decode(train_ids[:10].tolist())  # gegarandeerd geldig, uit de data zelf

    train_losses = []
    test_stappen, test_losses = [], []
    print(f"\ntrainen n_lagen={n_lagen} n_embed={n_embed} lengte={lengte} positie={gebruik_positie} feedforward={gebruik_feedforward} losse_qk={losse_qk} losse_v={losse_v} layernorm={gebruik_layernorm} masker={gebruik_masker} dropout={dropout} ({n_stappen} stappen)...")
    for stap in range(n_stappen):
        x_b, y_b = maak_batch(train_ids, lengte=lengte, aantal=aantal_train)
        scores_b, _ = model(x_b)
        loss_b = F.cross_entropy(scores_b.reshape(-1, tokenizer.vocab_size), y_b.reshape(-1))
        train_losses.append(loss_b.item())

        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()
        scheduler.step()

        if stap % eval_interval == 0 or stap == n_stappen - 1:
            model.eval()  # dropout uit tijdens meten, anders meet je een willekeurig uitgedund model
            with torch.no_grad():
                x_t, y_t = maak_batch(test_ids, lengte=lengte, aantal=aantal_test)
                scores_t, _ = model(x_t)
                loss_t = F.cross_entropy(scores_t.reshape(-1, tokenizer.vocab_size), y_t.reshape(-1))
            model.train()
            test_stappen.append(stap)
            test_losses.append(loss_t.item())
            huidige_lr = optimizer.param_groups[0]["lr"]
            sample = genereer(model, tokenizer, start=genereer_start, n_nieuw=40, lengte=lengte)
            print(f"  stap {stap:>5}  train loss {loss_b.item():.3f}  test loss {loss_t.item():.3f}  lr {huidige_lr:.5f}  sample: {sample!r}")

    return model, train_losses, test_stappen, test_losses


if __name__ == "__main__":
    tekst = TEKST_PAD.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(tekst)

    print(f"aantal karakters: {len(tekst)}")
    print(f"vocab size:       {tokenizer.vocab_size}")
    print(f"vocabulaire:      {''.join(tokenizer.chars)!r}")

    voorbeeld = tekst[:40]
    ids = tokenizer.encode(voorbeeld)
    print(f"\nvoorbeeld: {voorbeeld!r}")
    print(f"encoded:   {ids}")
    print(f"decoded:   {tokenizer.decode(ids)!r}")

    # controle: encode -> decode moet de originele tekst opleveren
    ids_alles = torch.tensor(tokenizer.encode(tekst), dtype=torch.long)
    assert tokenizer.decode(ids_alles.tolist()) == tekst
    print("\nround-trip check ok")
    print(f"data tensor:      {tuple(ids_alles.shape)} {ids_alles.dtype}")

    # willekeurige stukjes van LENGTE karakters, met bijbehorende targets
    g = torch.Generator().manual_seed(123)
    x, y = maak_batch(ids_alles, lengte=LENGTE, aantal=5, generator=g)
    print(f"\nx: {tuple(x.shape)}   y: {tuple(y.shape)}")
    for xi, yi in zip(x, y):
        print(f"\n  input : {tokenizer.decode(xi.tolist())!r}")
        print(f"  target: {tokenizer.decode(yi.tolist())!r}")

    # zo ziet het voorspel-probleem er per positie uit
    print("\nvoorspel-paren van het eerste stukje:")
    for i in range(5):
        context = tokenizer.decode(x[0, :i + 1].tolist())
        volgende = tokenizer.int_naar_char[y[0, i].item()]
        print(f"  {context!r:>28}  ->  {volgende!r}")

    # affiniteitsmatrix: één gedeelde matrix i.p.v. losse Q/K, per letterpaar
    print("\naffiniteits-model (ongetraind, n_embed=12, n_lagen=1) op het eerste stukje:")
    aff_model = AffiniteitsModel(tokenizer.vocab_size, n_embed=12, n_lagen=1)
    scores, gewichten = aff_model(x[:1])  # alleen het eerste stukje
    letters = [tokenizer.int_naar_char[t] for t in x[0].tolist()]
    for i in range(1, len(letters)):
        w = gewichten[0, i, :i + 1]
        top = torch.topk(w, k=min(3, i + 1))
        onderdelen = ", ".join(
            f"{letters[j]!r}:{p:.2f}" for p, j in zip(top.values.tolist(), top.indices.tolist())
        )
        context = "".join(letters[:i + 1])
        print(f"  positie {i:>2}  ({letters[i]!r} na {context!r})  let vooral op: {onderdelen}")
    print(f"\nscores (voorspelling per positie): {tuple(scores.shape)}")

    # ongetrainde loss, puur als sanity check dat voorspelling + doel matchen
    loss = F.cross_entropy(scores[0], y[0])
    max_loss = torch.log(torch.tensor(float(tokenizer.vocab_size)))
    print(f"loss op het eerste stukje (ongetraind): {loss.item():.3f}  (willekeurig gokken: {max_loss:.3f})")

    # train/test-split: nooit trainen en meten op dezelfde tekst
    split = int(TRAIN_FRACTIE * len(ids_alles))
    train_ids, test_ids = ids_alles[:split], ids_alles[split:]
    print(f"\ntrain: {len(train_ids)} tekens, test: {len(test_ids)} tekens")

    # config hierboven vast, nu vergelijken: embedding-grootte
    # (eerder getest met de primitieve architectuur, nu opnieuw met residu/QKV/LayerNorm)
    resultaten = {}
    for n_embed in N_EMBED_OPTIES:
        _, train_losses, test_stappen, test_losses = train_affiniteitsmodel(
            n_lagen=N_LAGEN, n_embed=n_embed, train_ids=train_ids, test_ids=test_ids, tokenizer=tokenizer,
            lengte=LENGTE, gebruik_positie=GEBRUIK_POSITIE, gebruik_feedforward=GEBRUIK_FEEDFORWARD,
            losse_qk=LOSSE_QK, losse_v=LOSSE_V,
            gebruik_layernorm=GEBRUIK_LAYERNORM, gebruik_masker=GEBRUIK_MASKER, dropout=DROPOUT,
            aantal_train=BATCH_AANTAL, aantal_test=TEST_BATCH_AANTAL,
            lr=LEERRATE, n_stappen=N_STAPPEN, eval_interval=EVAL_INTERVAL, seed=SEED,
        )
        resultaten[n_embed] = (train_losses, test_stappen, test_losses)

    # plotje: train/test loss per embedding-grootte, in dezelfde kleur per model
    plt.figure(figsize=(9, 6))
    kleuren = plt.cm.tab10.colors
    for i, n_embed in enumerate(N_EMBED_OPTIES):
        train_losses, test_stappen, test_losses = resultaten[n_embed]
        kleur = kleuren[i % len(kleuren)]
        plt.plot(range(len(train_losses)), train_losses, color=kleur, alpha=0.25)
        plt.plot(test_stappen, test_losses, color=kleur, marker="o", label=f"n_embed={n_embed}")
    plt.axhline(max_loss.item(), color="gray", linestyle="--", label="willekeurig gokken")
    plt.xlabel("stap")
    plt.ylabel("loss")
    plt.title(f"train/test loss per embedding-grootte (n_lagen={N_LAGEN}, lengte={LENGTE}, vaag = train, stippen = test)")
    plt.legend()
    plt.tight_layout()
    plot_pad = Path(__file__).parent / "loss.png"
    plt.savefig(plot_pad)
    print(f"\nplot opgeslagen: {plot_pad}")
