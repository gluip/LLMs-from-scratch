"""Maak DBNL-downloads schoon: alleen de lopende tekst overhouden.

    ../.venv/bin/python schoonmaak.py

Leest alles uit data/ruw/ en schrijft het resultaat naar data/.
Een DBNL-bestand begint met een blok bibliotheek-informatie (colofon,
verantwoording, rechthebbenden) en bevat door de tekst heen markeringen voor
paginanummers en illustraties. Dat is allemaal geen verhaal, dus het model
moet het niet te zien krijgen.
"""
import re
import unicodedata
from pathlib import Path

RUW = Path(__file__).parent / "data" / "ruw"
DOEL = Path(__file__).parent / "data"

# de laatste regel van het bibliotheek-voorwerk; alles daarvoor gaat weg
GRENS = "In dit bestand zijn twee typen markeringen opgenomen"

# markeringen als {==13==} {>>pagina-aanduiding<<}
MARKERING = re.compile(r"\{==.*?==\}\s*\{>>.*?<<\}", re.DOTALL)


def maak_schoon(tekst):
    regels = tekst.split("\n")
    for i, regel in enumerate(regels):
        if GRENS in regel:
            tekst = "\n".join(regels[i + 1:])
            break

    tekst = MARKERING.sub("", tekst)
    # losse {==...==} zonder bijbehorend {>>...<<} komen ook voor
    tekst = re.sub(r"\{==.*?==\}", "", tekst, flags=re.DOTALL)
    tekst = re.sub(r"\{>>.*?<<\}", "", tekst, flags=re.DOTALL)

    # harde spatie en tab zijn andere karakters dan een gewone spatie: het model
    # zou ze als los teken moeten leren terwijl ze hetzelfde betekenen
    tekst = tekst.replace("\xa0", " ").replace("\t", " ")
    tekst = unicodedata.normalize("NFC", tekst)

    # inspringing weghalen en regels met alleen spaties leegmaken
    regels = [regel.strip() for regel in tekst.split("\n")]
    tekst = "\n".join(regels)

    # meer dan één lege regel achter elkaar terugbrengen tot één
    tekst = re.sub(r"\n{3,}", "\n\n", tekst)
    return tekst.strip() + "\n"


if __name__ == "__main__":
    for bron in sorted(RUW.glob("*.txt")):
        ruw = bron.read_text(encoding="utf-8")
        schoon = maak_schoon(ruw)
        doel = DOEL / bron.name
        doel.write_text(schoon, encoding="utf-8")
        print(f"{bron.name:28s} {len(ruw):>9,d} -> {len(schoon):>9,d} karakters"
              f"  ({len(set(schoon)):d} unieke)")
