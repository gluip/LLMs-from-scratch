# dit is een experiment in een ander soort woordvoorspeller
# we trainen op de the verdict en proberen aldoor het
# maak een tokeniner op basis van the-verdict.txt

from pathlib import Path

import torch
import torch.nn as nn

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


class AffiniteitsModel(nn.Module):
    """Eén affiniteitsmatrix tussen alle letterparen (attention, minimaal).

    Elke letter krijgt een n_embed-dimensionale vector. In plaats van aparte
    Q- en K-matrices (zoals bij echte attention) gebruiken we hier maar één
    geleerde matrix W: elke embedding wordt ermee geprojecteerd, en daarna
    is de affiniteit tussen letter i en j simpelweg het inproduct van hun
    projecties. Hoog inproduct = de vectors wijzen dezelfde kant op = hoge
    affiniteit. Een causaal masker zorgt dat positie i alleen naar i en
    eerder mag kijken (je mag het antwoord niet vooruit zien).

    De V (value) is voorlopig nog gelijk aan diezelfde geprojecteerde g —
    later los te trekken in een eigen matrix, net als W nu al apart is van
    de ruwe embedding.
    """

    def __init__(self, vocab_size, n_embed=12):
        super().__init__()
        self.n_embed = n_embed
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.W = nn.Linear(n_embed, n_embed, bias=False)  # de ene gedeelde matrix

    def forward(self, x):
        T = x.shape[1]
        emb = self.embed(x)          # (batch, T, n_embed)
        g = self.W(emb)               # (batch, T, n_embed), geprojecteerd
        affiniteit = g @ g.transpose(-2, -1)          # (batch, T, T), inproduct per paar
        affiniteit = affiniteit / self.n_embed ** 0.5  # schaling, zoals bij echte attention

        masker = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        affiniteit = affiniteit.masked_fill(masker, float("-inf"))  # niet vooruitkijken

        gewichten = torch.softmax(affiniteit, dim=-1)  # per rij: kansen die optellen tot 1
        v = g  # de value is nu nog dezelfde g als hierboven, nog niet losgetrokken
        output = gewichten @ v  # (batch, T, n_embed), gewogen mix van de v's
        return affiniteit, gewichten, output


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
    print("\naffiniteits-model (ongetraind, n_embed=12) op het eerste stukje:")
    aff_model = AffiniteitsModel(tokenizer.vocab_size, n_embed=12)
    affiniteit, gewichten, output = aff_model(x[:1])  # alleen het eerste stukje
    letters = [tokenizer.int_naar_char[t] for t in x[0].tolist()]
    for i in range(1, len(letters)):
        w = gewichten[0, i, :i + 1]
        top = torch.topk(w, k=min(3, i + 1))
        onderdelen = ", ".join(
            f"{letters[j]!r}:{p:.2f}" for p, j in zip(top.values.tolist(), top.indices.tolist())
        )
        context = "".join(letters[:i + 1])
        print(f"  positie {i:>2}  ({letters[i]!r} na {context!r})  let vooral op: {onderdelen}")
    print(f"\noutput (gewogen mix per positie): {tuple(output.shape)}")
