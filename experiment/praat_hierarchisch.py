"""Prompt het hiërarchische chars->woord->transformer->chars model (zie
hierarchisch.py, EXPERIMENTEN.md experiment 14), zonder opnieuw te trainen.
Zusje van praat.py - zelfde REPL, maar praat met model_hierarchisch.pt in
plaats van model.pt, zodat je de twee naast elkaar kunt vergelijken.

    praat_hierarchisch.cmd                       # interactief
    praat_hierarchisch.cmd "wie is pinkeltje?"    # eenmalig, voor in een script

Of zonder de wrapper: ..\\.venv\\Scripts\\python.exe praat_hierarchisch.py

Dit model genereert per stap een heel WOORD (letter voor letter, tot het zelf
een stopteken voorspelt), niet één karakter. /n telt dus woorden, niet
karakters - anders dan bij praat.py.
"""
import sys
from pathlib import Path

import torch

from exp import APPARAAT, CharTokenizer
from hierarchisch import HierarchischModel, splits_in_brokken, codeer_brok, genereer_hierarchisch

MODEL_PAD = Path(__file__).parent / "model_hierarchisch.pt"
N_NIEUW = 40          # hoeveel woorden erbij
TEMPERATUUR = 0.7     # onder 1 = braver, boven 1 = wilder (zie EXPERIMENTEN.md: 0,6 gaf 100% bestaande
                      # woorden, 1,0 zakte naar 78% - dit model is gevoeliger voor temperatuur dan het char-model


def laad():
    if not MODEL_PAD.exists():
        sys.exit(f"{MODEL_PAD} bestaat niet — draai eerst train_hierarchisch.py om te trainen.")
    bundel = torch.load(MODEL_PAD, weights_only=False, map_location="cpu")
    cfg = bundel["config"]
    tokenizer = CharTokenizer("")
    tokenizer.chars = bundel["chars"]
    tokenizer.char_naar_int = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.int_naar_char = {i: ch for ch, i in tokenizer.char_naar_int.items()}
    model = HierarchischModel(
        tok_vocab_size=cfg["tok_vocab_size"], pad_id=cfg["pad_id"], eow_id=cfg["eow_id"],
        max_brok_lengte=cfg["max_brok_lengte"], n_embed_binnen=cfg["n_embed_binnen"],
        n_lagen_enc=cfg["n_lagen_enc"], n_lagen_dec=cfg["n_lagen_dec"], n_koppen_binnen=cfg["n_koppen_binnen"],
        n_embed_buiten=cfg["n_embed_buiten"], n_lagen_buiten=cfg["n_lagen_buiten"],
        n_koppen_buiten=cfg["n_koppen_buiten"], brok_venster=cfg["brok_venster"], dropout=0.0,
    )
    model.load_state_dict(bundel["state_dict"])
    model.eval().to(APPARAAT)
    return model, tokenizer, cfg["brok_venster"]


def schoon(tekst, tokenizer):
    """Onbekende karakters weghalen; codeer_brok zou er anders op stuklopen."""
    bekend = "".join(ch for ch in tekst if ch in tokenizer.char_naar_int)
    kwijt = sorted(set(tekst) - set(bekend))
    if kwijt:
        print(f"  (niet in vocabulaire, weggelaten: {kwijt})")
    return bekend


def antwoord(model, tokenizer, prompt, brok_venster, n_nieuw, temperatuur):
    p = schoon(prompt, tokenizer)
    if not p:
        print("  (niets bruikbaars in die prompt)\n")
        return
    start_brokken = [codeer_brok(b, tokenizer, model.max_brok_lengte, model.eow_id, model.pad_id)
                      for b in splits_in_brokken(p)]
    volledige_tekst, _ = genereer_hierarchisch(model, tokenizer, start_brokken, brok_venster,
                                                n_nieuw, APPARAAT, temperatuur=temperatuur)
    # net als praat.py: de | markeert waar het model begon. Bewust ASCII (cp1252-console).
    print(f"\n{p}|{volledige_tekst[len(p):]}\n{'-' * 70}")


def interactief(model, tokenizer, brok_venster):
    n_nieuw, temperatuur = N_NIEUW, TEMPERATUUR
    print(f"hierarchisch model geladen op {APPARAAT} (venster {brok_venster} brokken, "
          f"max_brok_lengte {model.max_brok_lengte})")
    print(f"typ een prompt, of /temp <getal>, /n <getal>, /help, /quit\n"
          f"nu: {n_nieuw} woorden, temperatuur {temperatuur}\n")
    while True:
        try:
            regel = input("> ").lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not regel:
            continue
        if regel in ("/quit", "/exit", "/q"):
            return
        if regel == "/help":
            print("  /temp <getal>  lager = braver, hoger = wilder (nu {:.2f})".format(temperatuur))
            print("  /n <getal>     hoeveel WOORDEN erbij (nu {})".format(n_nieuw))
            print("  /quit          stoppen\n")
            continue
        if regel.startswith("/temp"):
            try:
                temperatuur = float(regel.split()[1])
                print(f"  temperatuur = {temperatuur}\n")
            except (IndexError, ValueError):
                print("  gebruik: /temp 0.7\n")
            continue
        if regel.startswith("/n"):
            try:
                n_nieuw = int(regel.split()[1])
                print(f"  {n_nieuw} woorden\n")
            except (IndexError, ValueError):
                print("  gebruik: /n 40\n")
            continue
        if regel.startswith("/"):
            print("  onbekend commando, /help voor de lijst\n")
            continue
        antwoord(model, tokenizer, regel, brok_venster, n_nieuw, temperatuur)


if __name__ == "__main__":
    model, tokenizer, brok_venster = laad()
    if len(sys.argv) > 1:
        for prompt in sys.argv[1:]:
            print(f"\n>>> {prompt!r}")
            antwoord(model, tokenizer, prompt, brok_venster, N_NIEUW, TEMPERATUUR)
    else:
        interactief(model, tokenizer, brok_venster)
