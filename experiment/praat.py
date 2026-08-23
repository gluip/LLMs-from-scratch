"""Prompt het getrainde model uit exp.py, zonder opnieuw te trainen.

    ../.venv/bin/python praat.py "wie is pinkeltje?"
    ../.venv/bin/python praat.py            # gebruikt een paar vaste voorbeelden

Let op wat dit model wel en niet is: het is getraind om het volgende karakter
te voorspellen in de tekst van één kinderboek. Het heeft nooit een vraag met
een antwoord erachter gezien, dus het beantwoordt je vraag niet — het zet hem
voort alsof het een zin uit het boek is.
"""
import sys
from pathlib import Path

import torch

from exp import AffiniteitsModel, CharTokenizer, genereer

MODEL_PAD = Path(__file__).parent / "model.pt"


def laad():
    if not MODEL_PAD.exists():
        sys.exit(f"{MODEL_PAD} bestaat niet — draai eerst exp.py om te trainen.")
    bundel = torch.load(MODEL_PAD, weights_only=False)
    tokenizer = CharTokenizer("")
    tokenizer.chars = bundel["chars"]
    tokenizer.char_naar_int = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.int_naar_char = {i: ch for ch, i in tokenizer.char_naar_int.items()}
    model = AffiniteitsModel(tokenizer.vocab_size, **bundel["config"])
    model.load_state_dict(bundel["state_dict"])
    model.eval()
    return model, tokenizer, bundel["lengte"]


def schoon(tekst, tokenizer):
    """Onbekende karakters weghalen; het vocabulaire is maar 69 tekens groot."""
    bekend = "".join(ch for ch in tekst if ch in tokenizer.char_naar_int)
    kwijt = sorted(set(tekst) - set(bekend))
    if kwijt:
        print(f"  (niet in vocabulaire, weggelaten: {kwijt})")
    return bekend


if __name__ == "__main__":
    model, tokenizer, lengte = laad()
    prompts = sys.argv[1:] or [
        "wie is pinkeltje?",
        "Pinkeltje is ",
        "Op een dag ",
    ]
    for prompt in prompts:
        p = schoon(prompt, tokenizer)
        if not p:
            print(f"{prompt!r} -> niets bruikbaars over\n")
            continue
        uit = genereer(model, tokenizer, start=p, n_nieuw=120, lengte=lengte)
        print(f"\n>>> {prompt!r}")
        print(f"{uit}\n{'-' * 70}")
