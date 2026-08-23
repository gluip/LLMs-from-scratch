# dit is een experiment in een ander soort woordvoorspeller
# we trainen op de the verdict en proberen aldoor het
# maak een tokeniner op basis van the-verdict.txt

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

TEKST_PAD = Path(__file__).parent / "pinkeltje_schoon.txt"


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

    In plaats van aparte Q- en K-matrices (zoals bij echte attention)
    gebruiken we hier maar één geleerde matrix W: elke vector wordt ermee
    geprojecteerd, en de affiniteit tussen positie i en j is simpelweg het
    inproduct van hun projecties. Hoog inproduct = de vectors wijzen dezelfde
    kant op = hoge affiniteit. Een causaal masker zorgt dat positie i alleen
    naar i en eerder mag kijken (je mag het antwoord niet vooruit zien).

    De V (value) is voorlopig nog gelijk aan diezelfde geprojecteerde g —
    later los te trekken in een eigen matrix, net als W nu al apart is van
    de ruwe input.
    """

    def __init__(self, n_embed):
        super().__init__()
        self.n_embed = n_embed
        self.W = nn.Linear(n_embed, n_embed, bias=False)  # de ene gedeelde matrix

    def forward(self, h):
        # h: (batch, T, n_embed) -> ook weer (batch, T, n_embed), dus stackbaar
        T = h.shape[1]
        g = self.W(h)               # (batch, T, n_embed), geprojecteerd
        affiniteit = g @ g.transpose(-2, -1)           # (batch, T, T), inproduct per paar
        affiniteit = affiniteit / self.n_embed ** 0.5   # schaling, zoals bij echte attention

        masker = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        affiniteit = affiniteit.masked_fill(masker, float("-inf"))  # niet vooruitkijken

        gewichten = torch.softmax(affiniteit, dim=-1)  # per rij: kansen die optellen tot 1
        v = g  # de value is nu nog dezelfde g als hierboven, nog niet losgetrokken
        output = gewichten @ v  # (batch, T, n_embed), gewogen mix van de v's
        return output, gewichten


class AffiniteitsModel(nn.Module):
    """Embedding -> n_lagen stackbare AffiniteitsLaag'en -> voorspelling per letter."""

    def __init__(self, vocab_size, n_embed=12, n_lagen=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.lagen = nn.ModuleList([AffiniteitsLaag(n_embed) for _ in range(n_lagen)])
        self.uit = nn.Linear(n_embed, vocab_size)  # enige plek die naar vocab_size gaat

    def forward(self, x):
        h = self.embed(x)  # (batch, T, n_embed)
        for laag in self.lagen:
            h, gewichten = laag(h)  # zelfde vorm erin als eruit
        scores = self.uit(h)  # (batch, T, vocab_size), pas hier naar vocab_size
        return scores, gewichten  # gewichten van de laatste laag, voor inspectie


def genereer(model, tokenizer, start, n_nieuw=40, generator=None):
    """Genereer karakter voor karakter verder op `start`, door telkens uit de
    voorspelde kansverdeling te samplen (niet steeds de meest waarschijnlijke)."""
    model.eval()
    ids = torch.tensor([tokenizer.encode(start)], dtype=torch.long)  # (1, T)
    with torch.no_grad():
        for _ in range(n_nieuw):
            scores, _ = model(ids)
            kansen = torch.softmax(scores[0, -1], dim=-1)  # kansen voor het laatste teken
            volgend = torch.multinomial(kansen, num_samples=1, generator=generator)
            ids = torch.cat([ids, volgend.unsqueeze(0)], dim=1)
    model.train()
    return tokenizer.decode(ids[0].tolist())


def train_affiniteitsmodel(n_lagen, train_ids, test_ids, tokenizer, n_stappen=3000, eval_interval=300, seed=0):
    """Traint een AffiniteitsModel met `n_lagen` lagen en houdt de loss bij."""
    torch.manual_seed(seed)
    model = AffiniteitsModel(tokenizer.vocab_size, n_embed=12, n_lagen=n_lagen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    train_losses = []
    test_stappen, test_losses = [], []
    print(f"\ntrainen n_lagen={n_lagen} ({n_stappen} stappen)...")
    for stap in range(n_stappen):
        x_b, y_b = maak_batch(train_ids, lengte=20, aantal=64)
        scores_b, _ = model(x_b)
        loss_b = F.cross_entropy(scores_b.reshape(-1, tokenizer.vocab_size), y_b.reshape(-1))
        train_losses.append(loss_b.item())

        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()

        if stap % eval_interval == 0 or stap == n_stappen - 1:
            with torch.no_grad():
                x_t, y_t = maak_batch(test_ids, lengte=20, aantal=256)
                scores_t, _ = model(x_t)
                loss_t = F.cross_entropy(scores_t.reshape(-1, tokenizer.vocab_size), y_t.reshape(-1))
            test_stappen.append(stap)
            test_losses.append(loss_t.item())
            sample = genereer(model, tokenizer, start="Pinkeltje ", n_nieuw=40)
            print(f"  stap {stap:>5}  train loss {loss_b.item():.3f}  test loss {loss_t.item():.3f}  sample: {sample!r}")

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

    # willekeurige stukjes van 20 karakters, met bijbehorende targets
    g = torch.Generator().manual_seed(123)
    x, y = maak_batch(ids_alles, lengte=20, aantal=5, generator=g)
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
    split = int(0.9 * len(ids_alles))
    train_ids, test_ids = ids_alles[:split], ids_alles[split:]
    print(f"\ntrain: {len(train_ids)} tekens, test: {len(test_ids)} tekens")

    # train meerdere modellen met een verschillend aantal lagen, om te vergelijken
    n_lagen_opties = [1, 2, 3, 4]
    resultaten = {}
    for n_lagen in n_lagen_opties:
        _, train_losses, test_stappen, test_losses = train_affiniteitsmodel(
            n_lagen, train_ids, test_ids, tokenizer, n_stappen=3000, eval_interval=300,
        )
        resultaten[n_lagen] = (train_losses, test_stappen, test_losses)

    # plotje: train/test loss per aantal lagen, in dezelfde kleur per model
    plt.figure(figsize=(9, 6))
    kleuren = plt.cm.tab10.colors
    for i, n_lagen in enumerate(n_lagen_opties):
        train_losses, test_stappen, test_losses = resultaten[n_lagen]
        kleur = kleuren[i % len(kleuren)]
        plt.plot(range(len(train_losses)), train_losses, color=kleur, alpha=0.25)
        plt.plot(test_stappen, test_losses, color=kleur, marker="o", label=f"n_lagen={n_lagen}")
    plt.axhline(max_loss.item(), color="gray", linestyle="--", label="willekeurig gokken")
    plt.xlabel("stap")
    plt.ylabel("loss")
    plt.title("train/test loss per aantal lagen (vaag = train, stippen = test)")
    plt.legend()
    plt.tight_layout()
    plot_pad = Path(__file__).parent / "loss.png"
    plt.savefig(plot_pad)
    print(f"\nplot opgeslagen: {plot_pad}")
