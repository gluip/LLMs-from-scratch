# Handoff — stand van zaken

Laatst bijgewerkt: 28 augustus 2026. Branch `feature/experiments`.

Er lopen twee sporen in deze repo, en ze zijn recent bij elkaar gekomen.

| spoor | map | waar het over gaat |
|---|---|---|
| taalmodel | `experiment/` | char-level transformer op 14 Nederlandse boeken (10,2M karakters) |
| rekenmodel | `exp-math/` | hetzelfde soort model op optellen, aftrekken en vermenigvuldigen |

Elk spoor heeft zijn eigen `EXPERIMENTEN.md` met de volledige chronologie.
**Begin daar**, niet bij dit bestand — `exp-math/EXPERIMENTEN.md` heeft
bovenaan een uitgebreidere "Huidige staat" met bestandsoverzicht en de
kleinste werkende configuratie per bewerking.

## Waar het nu staat

**Rekenspoor (`exp-math/`) — 15 experimenten, afgerond.** Kleinste model dat
foutloos is, per bewerking:

| bewerking | nodige onderdelen | parameters |
|---|---|---|
| optellen | geen (vaste middeling volstaat) | 49 |
| aftrekken | positie + aandacht + `tanh` | 153 |
| optellen én aftrekken | positie + aandacht + `W_o`, 4 koppen | 553 |
| alle drie (`+ - *`) | + feedforward met kwadratische activatie | 14.337 (100%) |

Vier verslagen in HTML, standalone en offline te lezen: `verslag.html`
(optellen), `verslag-aftrekken.html` (aftrekken), `verslag-machinerie.html`
(hoe het optelt) en `verslag-maal.html` (hoe het vermenigvuldigt, van de
grond af uitgelegd).

**Taalspoor (`experiment/`) — beste model staat op 1,2605** (laatste-positie-
loss, `model.pt`). Daar is deze sessie niets aan veranderd; wel is
`vorm_vgl.py` toegevoegd, zie hieronder.

## Wat er als laatste gebeurde, en waar het bleef liggen

De laatste vraag was of een netwerk zijn eigen vorm kan leren: nodes snoeien,
nodes laten aangroeien, en elke node zijn eigen activatiefunctie laten kiezen.
Dat is eerst op het rekenmodel gedaan (experiment 13–15 in
`exp-math/EXPERIMENTEN.md`) en daarna overgehaald naar het taalmodel
(`experiment/vorm_vgl.py`).

**De uitkomsten spreken elkaar tegen, en dat is het interessante.**

Op het rekenmodel kwam alles binnen de ruis gelijk uit — omdat die taak te
makkelijk is: 4 tot 32 verborgen eenheden geven allemaal 93–97%, dus het maakt
niet uit wélke je kiest. Alleen snoeien-vanaf-groot was aantoonbaar slecht
(87% tegen 97% voor vanaf nul trainen, en 30.000 extra stappen halen dat niet
in).

Op het taalmodel is het beeld wél scherp, en omgekeerd: bij **gelijk aantal
parameters** verslaat een smal-maar-dicht model beide dunne varianten met 0,09
nats — vier tot negen keer de ruismarge.

| aanpak (taalmodel, 6000 stappen) | loss |
|---|---|
| dynamisch groeien/afsterven | 1,3949 |
| vast willekeurig deel | 1,4037 |
| **dicht model op hetzelfde budget** | **1,3073** |

### Wat er open ligt

**1. De activatie-vergelijking op taal is niet af.** De run is afgebroken na
drie van de vijf condities:

| activatie | loss |
|---|---|
| relu | 1,2298 |
| gelu | 1,2180 |
| kwadraat | 1,2297 |

Nog te draaien: `lineair`, en vooral **vrije keuze per eenheid** — waarbij elke
eenheid zelf een mengsel van de vier functies kiest en je achteraf kunt
uitlezen waarvoor taal kiest. Bij het rekenmodel koos 74–96 van de 128
eenheden het kwadraat (logisch, want `a*b = ((a+b)²-(a-b)²)/4`). De
vergelijking met taal is de eigenlijke vraag en die staat nog open.

`vorm_vgl.py` is GPU-klaar. Op een GPU horen `N_STAPPEN` op 18000 en `SEEDS`
op `range(3)`; nu staan ze op 6000 en 2 omdat één run op CPU ~4 minuten kost.

**2. Twee dingen die ik zelf niet heb afgemaakt**, allebei genoemd in
`exp-math/EXPERIMENTEN.md` experiment 15:

- De poorten-op-blokken opzet (leerbare diepte) **deugt niet**: `alfa` zakt
  netjes naar 0,1 maar het blok schaalt zijn interne gewichten evenredig op,
  dus `alfa` meet niets. Ablatie laat zien dat alle blokken dragend zijn. De
  reparatie is het blok normaliseren vóór de poort, zodat `alfa` de enige
  schaalknop is. Niet uitgevoerd.
- Bij vermenigvuldigen blijft het model op 98% steken (met ReLU; met een
  kwadratische activatie wel 100%). De resterende fouten zijn uitsluitend
  antwoorden vlak bij nul, en dat is een **bereikprobleem**: de antwoorden
  lopen van −9 tot 81 en er moet op een halve eenheid nauwkeurig afgerond
  worden. Te omzeilen door het antwoord anders te coderen, maar dat verandert
  de taak.

## Werkafspraken die in deze repo gelden

- **Draai met `python -u`.** Zonder dat buffert Python de uitvoer en zie je
  bij de langere sweeps minutenlang niets.
- **Eén run is geen resultaat.** Rapporteer over meerdere seeds, met de
  laagste erbij. In het rekenspoor bleek de uitkomst ooit tussen 40% en 100%
  te liggen op niets dan de startgewichten.
- **Ruismarge:** taalmodel 0,01–0,02 nats tussen identieke runs; rekenmodel
  5 procentpunt (20 testvragen).
- **Voeg na elk vergelijkingsscript een entry toe** aan de `EXPERIMENTEN.md`
  van dat spoor, onderaan, met opzet, uitkomst en conclusie.
- **Meet nooit op de data waarop je gefit hebt.** Bij het uitlezen van interne
  toestanden met een lineaire probe: 33 vrije parameters op 100 punten halen
  R² = +0,35 uit pure ruis. Kruisvalideren, met een ruis-doel als controle.
  Zie `exp-math/EXPERIMENTEN.md` experiment 12.

## Vergissingen die ik onderweg maakte

Staan uitgeschreven in de logboeken, omdat ze makkelijk te herhalen zijn:

- **Te ondiep gemeten en te vroeg geconcludeerd.** "Multi-head helpt niet bij
  aftrekken" kwam uit een test die bij 4 koppen ophield; bij 8 klapt hij om.
  Zie experiment 9a.
- **Twee dingen tegelijk veranderd.** "De feedforward levert 30 procentpunt
  door zijn niet-lineariteit" vergeleek *geen laag* met *laag mét ReLU*. Een
  lineaire laag haalt 99% en ReLU 97%, terwijl lineair geen uitdrukkingskracht
  toevoegt — het meeste kwam dus van de extra laag. Zie experiment 14.
- **Een maat gebruikt die niet meet wat je denkt.** R² van 0,986 klinkt bijna
  perfect maar komt neer op gemiddeld 2,35 mis, en er moet op een halve
  eenheid afgerond worden: 26% goed. Zie experiment 12.
