"""Prompt het getrainde model uit exp.py, zonder opnieuw te trainen.

    praat.cmd                       # interactief: typ een prompt, krijg tekst terug
    praat.cmd "wie is pinkeltje?"   # eenmalig, voor in een script

Of zonder de wrapper: ..\\.venv\\Scripts\\python.exe praat.py

Let op wat dit model wel en niet is: het is getraind om het volgende karakter
te voorspellen in de tekst van drie boeken. Het heeft nooit een vraag met een
antwoord erachter gezien, dus het beantwoordt je vraag niet — het zet hem voort
alsof het een zin uit een boek is. Verwacht Nederlands-klinkende woorden en af
en toe een kloppende zinswending, geen inhoud.
"""
import sys
from pathlib import Path

import torch

from exp import APPARAAT, AffiniteitsModel, CharTokenizer, genereer

MODEL_PAD = Path(__file__).parent / "model.pt"
N_NIEUW = 200        # hoeveel karakters erbij
TEMPERATUUR = 0.8    # onder 1 = braver, boven 1 = wilder


def laad():
    if not MODEL_PAD.exists():
        sys.exit(f"{MODEL_PAD} bestaat niet — draai eerst exp.py om te trainen.")
    bundel = torch.load(MODEL_PAD, weights_only=False, map_location="cpu")
    tokenizer = CharTokenizer("")
    tokenizer.chars = bundel["chars"]
    tokenizer.char_naar_int = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.int_naar_char = {i: ch for ch, i in tokenizer.char_naar_int.items()}
    model = AffiniteitsModel(tokenizer.vocab_size, **bundel["config"])
    model.load_state_dict(bundel["state_dict"])
    model.eval().to(APPARAAT)
    return model, tokenizer, bundel["lengte"]


def schoon(tekst, tokenizer):
    """Onbekende karakters weghalen; het vocabulaire is maar een stuk of 115 tekens."""
    bekend = "".join(ch for ch in tekst if ch in tokenizer.char_naar_int)
    kwijt = sorted(set(tekst) - set(bekend))
    if kwijt:
        print(f"  (niet in vocabulaire, weggelaten: {kwijt})")
    return bekend


def antwoord(model, tokenizer, prompt, lengte, n_nieuw, temperatuur):
    p = schoon(prompt, tokenizer)
    if not p:
        print("  (niets bruikbaars in die prompt)\n")
        return
    uit = genereer(model, tokenizer, start=p, n_nieuw=n_nieuw, lengte=lengte,
                   temperatuur=temperatuur)
    # de prompt zelf komt mee terug uit genereer; de | markeert waar het model begon.
    # Bewust ASCII: een Windows-console op cp1252 struikelt over fraaiere tekens.
    print(f"\n{p}|{uit[len(p):]}\n{'-' * 70}")


def interactief(model, tokenizer, lengte):
    n_nieuw, temperatuur = N_NIEUW, TEMPERATUUR
    print(f"model geladen op {APPARAAT} (venster {lengte}, vocab {tokenizer.vocab_size})")
    print(f"typ een prompt, of /temp <getal>, /n <getal>, /help, /quit\n"
          f"nu: {n_nieuw} karakters, temperatuur {temperatuur}\n")
    while True:
        try:
            # de ﻿ eraf: pipe je een bestand naar binnen, dan begint dat vaak
            # met een byte-order-mark, en die maakt van "/temp 0.5" een prompt
            regel = input("> ").lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):  # ctrl-D / ctrl-C is gewoon stoppen
            print()
            return
        if not regel:
            continue
        if regel in ("/quit", "/exit", "/q"):
            return
        if regel == "/help":
            print("  /temp <getal>  lager = braver, hoger = wilder (nu {:.2f})".format(temperatuur))
            print("  /n <getal>     hoeveel karakters erbij (nu {})".format(n_nieuw))
            print("  /quit          stoppen\n")
            continue
        if regel.startswith("/temp"):
            try:
                temperatuur = float(regel.split()[1])
                print(f"  temperatuur = {temperatuur}\n")
            except (IndexError, ValueError):
                print("  gebruik: /temp 0.8\n")
            continue
        if regel.startswith("/n"):
            try:
                n_nieuw = int(regel.split()[1])
                print(f"  {n_nieuw} karakters\n")
            except (IndexError, ValueError):
                print("  gebruik: /n 200\n")
            continue
        if regel.startswith("/"):
            print("  onbekend commando, /help voor de lijst\n")
            continue
        antwoord(model, tokenizer, regel, lengte, n_nieuw, temperatuur)


if __name__ == "__main__":
    model, tokenizer, lengte = laad()
    if len(sys.argv) > 1:  # prompts als argument: eenmalig, handig in een script
        for prompt in sys.argv[1:]:
            print(f"\n>>> {prompt!r}")
            antwoord(model, tokenizer, prompt, lengte, N_NIEUW, TEMPERATUUR)
    else:
        interactief(model, tokenizer, lengte)
