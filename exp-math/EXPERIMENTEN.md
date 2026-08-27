# Experimentenlogboek — wiskunde

Zelfde afspraak als `../experiment/EXPERIMENTEN.md`: **na elk nieuw
vergelijkingsscript een entry toevoegen** (onderaan, chronologisch), met
opzet, uitkomst en conclusie.

Dit spoor staat los van het char-level taalmodel in `../experiment/`.
`wiskunde.py` is daar een uitgeklede kopie van, maar evolueert apart.

---

## Huidige staat (bijgewerkt na experiment 11)

Kort overzicht om weer in te komen zonder het hele logboek te lezen.

### Wat er staat

| bestand | wat |
|---|---|
| `rekenen.py` | **het huidige model.** Eén klasse `Rekenmodel`, elk onderdeel een knop (`positie`, `leer_aandacht`, `ff`, `n_koppen`, `uit_proj`, `soort`). Plus de drie datasets en de zoektocht naar het minimale model. |
| `wiskunde.py` | het eerste optel-model (experiment 1 t/m 6). Blijft staan omdat `stabiliteit_vgl.py`, `embeddings_kijken.py` en `binnenkant.py` zijn knoppen gebruiken. |
| `uitleg.py` | rekent de hele keten na voor één som en maakt de figuren van `verslag-machinerie.html`. |
| `stabiliteit_vgl.py` | weight-decay-sweep (exp. 2). 90 runs, ~12 min. |
| `embeddings_kijken.py` | de getallenlijn en layernorm (exp. 3–4). |
| `binnenkant.py` | Q/K/V en de ablaties (exp. 5–6). |
| `vergelijk_bewerkingen.py` | optellen vs aftrekken, gewichtsvormen, koppen (exp. 7–10). |
| `data/` | `simple.txt` (optellen), `aftrekken.txt`, `vermenigvuldigen.txt`, `optellen_aftrekken.txt` |
| `verslag*.html` | drie standalone verslagen, figuren ingesloten, werken offline |

Draaien met `.venv/bin/python -u <script>`. **Gebruik `-u`**, anders buffert
Python de uitvoer en zie je minutenlang niets.

### Kleinste model dat foutloos is, per bewerking

| bewerking | nodige onderdelen | n_embed | parameters |
|---|---|---|---|
| optellen | geen (vaste middeling volstaat) | 2 | **49** |
| aftrekken | positie + aandacht + `soort="tanh"` | 4 | **153** |
| optellen én aftrekken | positie + aandacht + `W_o`, 4 koppen | 8 | **553** |
| alle drie (`+ - *`) | positie + aandacht + `W_o` + `ff`, 8 koppen | 32 | 14.337 &mdash; 98%, geen 100% |

De eerste drie zijn 100% bij elke seed; de gecombineerde taak met
vermenigvuldigen blijft op 98% steken (zie experiment 11 — dat is een
bereikprobleem, geen architectuurprobleem). Gemeenschappelijk:
`weight_decay=0,3`, `lr=3e-3` cosine, `batch=16`, splitsing 80/20 met
`SPLITS_SEED=42`. `n_stappen=10000`, behalve bij de drie bewerkingen: 30.000.

### De vier dingen die er echt toe doen

1. **Weight decay 0,3 is de belangrijkste knop** (exp. 2). Op AdamW's default
   varieerde de uitkomst van 40% tot 100% over niets dan de startgewichten.
2. **Layernorm uit** bij één laag (exp. 4). Met layernorm liggen de cijfers op
   een boog en haalt het model 95%; zonder wordt het een rechte lijn met
   gelijke stappen en 100%. Zet hem terug aan zodra je lagen toevoegt.
3. **De aandacht doet per taak iets anders** (exp. 5, 7, 10): middelen bij
   optellen, kiezen bij aftrekken, schakelen op het operator-teken bij beide.
4. **De taak bepaalt de architectuur.** Wat bij optellen weggelaten kon
   worden, is bij aftrekken het verschil tussen 9% en 100%. En de
   feedforward, die bij optellen én aftrekken overbodig was, scheelt bij
   vermenigvuldigen dertig procentpunt (exp. 11) — precies omdat dat de enige
   niet-lineaire bewerking van de drie is.

### Nog te doen

- **Het bereikprobleem bij drie bewerkingen** (exp. 11). Het model blijft op
  98% steken en de fouten zitten allemaal bij antwoorden vlak bij nul, omdat
  het bereik nu −9..81 is. Te omzeilen door het antwoord anders te coderen
  (cijfer voor cijfer, of een kop per bewerking) — maar dat verandert de taak.
- **Vermenigvuldigen apart.** `data/vermenigvuldigen.txt` (100 regels) is nog
  niet los gemeten, alleen in de gecombineerde set.
- **`tanh` bij lange reeksen.** Wint bij de losse bewerkingen maar faalt bij
  de gecombineerde taak (exp. 9–10), en er is niets dat de schaal beteugelt
  als T groeit. Vraagt om een taak met variabele lengte.
- **Opruimen.** `wiskunde.py` kan weg zodra de drie oude analyse-scripts naar
  `rekenen.py` zijn overgezet.

---

## De vraag

Niet "kan een transformer 100 regels onthouden" (dat kan hij), maar:

> **leert het netwerk optellen, of leert het de tabel uit z'n hoofd?**

Daarom wordt 20% van de sommen achtergehouden. Alleen de test-accuratesse
telt; train-accuratesse van 100% zegt niets.

## Vaste opzet van experiment 1 t/m 6

Deze gold voor het optel-spoor met `wiskunde.py`. Vanaf experiment 7 draait
alles op `rekenen.py`; zie "Huidige staat" hierboven voor de instellingen die
nu gelden. Tenzij een entry iets anders zegt:

`data/simple.txt` (100 regels, `a + b = c` voor a,b van 0 t/m 9),
tokenisatie op getalniveau (21 tokens, `18` is één token), venster 4,
splitsing 80/20 random met `SPLITS_SEED=42`, `n_lagen=1, n_embed=32,
n_koppen=4, ff_factor=4.0, batch_aantal=16, lr=3e-3 cosine, n_stappen=3000`,
10 seeds.

**Ruis:** met 20 testvragen springt de accuratesse in stappen van 5%, en de
spreiding tussen seeds is groot (zie experiment 1). Eén run zegt niets —
altijd over de 10 seeds rapporteren, met min en max erbij.

---

## 1. Kwadratisch verschil vs. cross-entropy (`wiskunde.py`)

**Opzet.** Dezelfde splitsing en dezelfde seeds, twee uitvoerkoppen:

| `LOSS_SOORT` | kop | loss | "goed" |
|---|---|---|---|
| `kwadratisch_verschil` | `Linear(n_embed, 1)` | `(voorspelling - antwoord)²` | `round(uitvoer) == antwoord` |
| `kruisentropie` | `Linear(n_embed, 21)` | cross-entropy over tokens | `argmax == antwoord` |

**Uitkomst.** Train-accuratesse is bij allebei 100% bij elke seed — de 80
trainingsregels worden altijd gememoriseerd. Op de achtergehouden 20:

| loss | test (gem.) | min | max | per seed |
|---|---|---|---|---|
| **kwadratisch verschil** | **84,0%** | 40% | 100% | 100, 40, 90, 100, 70, 95, 100, 75, 95, 75 |
| cross-entropy | 30,5% | 10% | 65% | 10, 30, 20, 15, 35, 50, 65, 30, 20, 30 |

**Conclusie.** Twee dingen, en het tweede is het interessantst.

1. De verschil-loss wint ruim (84% vs 30%), en dat is geen verrassing als je
   bedenkt wát er verschilt: die loss geeft het model de getallenlijn cadeau.
   `9+9=17` is een kleine fout en `9+9=3` een grote. Bij cross-entropy zijn
   beide precies even fout — "verkeerd token" — en moet het model de ordening
   van `0`..`18` zelf uit 80 voorbeelden afleiden. Dat lukt grotendeels niet.
   De vergelijking meet dus niet "welke loss is beter", maar **hoeveel het
   scheelt om de structuur van het antwoord cadeau te krijgen**.

2. De spreiding is enorm: 40% tot 100% over alleen de seed. Zelfde data,
   zelfde splitsing, zelfde hyperparameters — alleen andere startgewichten.
   Het model leert dus niet betrouwbaar optellen; het belandt afhankelijk van
   z'n initialisatie in een oplossing die wel of niet generaliseert. Een
   enkele run van 100% (zoals seed 0, alle 20 goed) zou zonder de andere
   seeds de verkeerde conclusie hebben opgeleverd.

Bij de goede runs zitten de ruwe uitvoeren opvallend dicht op het gehele
getal (`9+9` -> 17,91, `1+9` -> 10,01), dus het model rekent daar echt iets
uit in plaats van naar een cluster te gokken.

**Vervolgknoppen.** Een attention-laag erbij; `n_embed` omlaag om
memoriseren te bemoeilijken; meer data (twee cijfers); de embeddings van
`0`..`9` inspecteren om te zien of er een getallenlijn in zit — en of dat
verschilt tussen een seed die 100% haalt en een die 40% haalt.

---

## 2. Stabiliteit: weight decay (`stabiliteit_vgl.py`)

**Opzet.** Experiment 1 haalde gemiddeld 84% maar met een spreiding van 40%
tot 100% over niets dan de startgewichten. Een model dat bij een op de vijf
seeds instort is niet bruikbaar, dus dit experiment zoekt de knop die de
**ondergrens** omhoog haalt, niet het gemiddelde. Gesweept: weight decay ×
trainingsduur (5 seeds), daarna model-capaciteit bovenop de beste instelling.

**Uitkomst — weight decay × trainingsduur:**

| weight decay | 3000 stappen (gem / min) | 10000 stappen (gem / min) |
|---|---|---|
| 0,0 | 82% / 50% | 86% / 50% |
| 0,01 (AdamW default) | 80% / **40%** | 86% / 45% |
| 0,1 | 92% / 75% | 95% / 85% |
| **0,3** | 92% / 85% | **95% / 95%** |
| 1,0 | 90% / 90% | 94% / 90% |

**Uitkomst — capaciteit, bij wd=0,3 en 10000 stappen:**

| model | gem | min | max |
|---|---|---|---|
| n_lagen=1, n_embed=16 | 95% | 95% | 95% |
| n_lagen=1, n_embed=32 | 95% | 95% | 95% |
| n_lagen=1, n_embed=64 | 98% | 95% | 100% |
| n_lagen=2, n_embed=32 | 93% | 90% | 95% |

Finale, wd=0,3 en 10000 stappen over 10 seeds: **96% gemiddeld, min 95%,
max 100%**.

**Conclusie.**

1. **Weight decay is de knop, trainingsduur maakt hem af.** Van 0,01 naar 0,3
   gaat de ondergrens van 40% naar 85%; 3000 -> 10000 stappen tilt hem naar
   95%. Weight decay trekt gewichten naar 0 tenzij de data ze nodig heeft, en
   een tabel onthouden kost veel losse grote gewichten terwijl één regel die
   voor alle sommen werkt er minder kost. De straf op grootte maakt de regel
   dus goedkoper dan de tabel. Zonder weight decay memoriseert het model
   sowieso (train 100% binnen 250 stappen) en is het puur van de
   initialisatie afhankelijk of er daarnaast iets generaliseerbaars ontstaat.

2. **Meer capaciteit helpt niet, meer diepte schaadt licht.** n_embed 16, 32
   en 64 geven dezelfde ondergrens; een tweede attention-laag maakt het
   meetbaar slechter (min 90%). De taak past ruim in één laag.

3. **Het plafond van 95% is geen modelfout maar een eigenschap van de
   splitsing.** De enige fout is steeds `9 + 9 = 18`, bij 9 van de 10 seeds,
   en altijd met antwoord 17. Reden: `9+9` is de enige som die 18 oplevert en
   die zit in de testset, dus **18 komt nul keer voor in de trainingsdata**.
   De antwoorden in train lopen van 0 t/m 17. Voor een regressie-kop is 18
   vragen daarmee extrapolatie voorbij het bereik dat hij ooit heeft moeten
   produceren — 17 antwoorden is dan precies wat je verwacht. 19/20 is op
   deze splitsing feitelijk foutloos.

   Dat n_embed=64 het bij 3 van de 5 seeds tóch goed kreeg is het enige
   teken dat die extrapolatie niet principieel onmogelijk is.

**Kosten.** 90 trainingsruns, ~12 minuten op de CPU. Let op bij het draaien:
gebruik `python -u`, anders buffert Python de uitvoer tot het eind en zie je
tien minuten lang niets.

**Vervolgknoppen.** De randen van het bereik zijn nu de bottleneck, dus:
een splitsing die niet toevallig het enige exemplaar van een antwoord
achterhoudt (of erover rapporteren per antwoord-frequentie); twee cijfers
zodat elk antwoord vaker voorkomt; en de embeddings van `0`..`9`
inspecteren om te zien of er een getallenlijn in zit — nu met een config die
betrouwbaar generaliseert, dus dat is een eerlijke vergelijking geworden.

---

## 3. Zit er semantiek in de gewichten? (`embeddings_kijken.py`)

**Opzet.** De cijfers `0`..`9` komen binnen als losse symbolen; niets vertelt
het model dat 7 groter is dan 3. Als die ordening tóch in de geleerde
embeddings terugkomt, heeft het model iets over getallen begrepen. Per seed
de embedding-matrix van `0`..`9` genomen (10 x 32), gecentreerd, en met de
**volledige SVD** op zijn hoofdassen geprojecteerd.

> Valkuil: `torch.pca_lowrank(q=2)` geeft per definitie 100% verklaarde
> variantie terug, want het kent maar twee componenten. Dan meet je je eigen
> keuze. Gebruik `torch.linalg.svd`.

**Uitkomst** (config van experiment 2, dus mét layernorm), 6 seeds:

| maat | waarde |
|---|---|
| variantie in PC1 | 52–59% |
| variantie in PC2 | 27–32% |
| PC1 + PC2 samen | **82–86%** |
| correlatie hoek langs de boog ↔ getalwaarde | **+0,994 … +0,997** |

**Conclusie.** Er zit semantiek in. De 32 dimensies klappen in tot een plat
vlak, en daarin liggen de cijfers op volgorde langs een **boog**, met de hoek
evenredig aan de getalwaarde. Bij elke seed dezelfde vorm. De afstand
`|e(a)-e(b)|` groeit met `|a-b|` maar vlakt af rond 5 — precies wat een boog
doet en een rechte lijn niet.

Meegenomen controle: de antwoord-tokens `10`..`18` zijn bij deze loss nooit
invoer (het antwoord is een getal, geen token), dus ze krijgen nooit
gradiënt. Weight decay drukt ze plat: gemiddelde norm 0,071 tegen 0,170 voor
de cijfers die wel gebruikt worden. Zichtbaar bewijs dat weight decay werkt.

---

## 4. Layernorm uit: de boog wordt een liniaal (`embeddings_kijken.py`)

**Opzet.** Experiment 3 riep de vraag op *waarom* het een boog is en geen
rechte lijn. Hypothese: layernorm normaliseert de lengte van elke vector weg,
dus kan alleen de **richting** nog informatie dragen — en richtingen liggen
op een cirkel. Toets: `gebruik_layernorm=False`. Dat was al een knop in
`SomModel`, er hoefde niets aangepast.

**Uitkomst**, 10 seeds, verder identieke config:

| | test (gem / min) | PC1 | PC2 | PC1 ↔ waarde |
|---|---|---|---|---|
| mét layernorm | 95% / 95% | 56% | 29% | 0,945 (boog) |
| **zonder layernorm** | **100% / 100%** | **99%** | **1%** | **1,000 (recht)** |

Stapgrootte tussen opeenvolgende cijfers op PC1: 0,2062 ± 0,0040 — **1,9%
variatie**. Dat is geen lijn maar een liniaal.

**Conclusie.** De hypothese klopt, en het levert meer op dan een mooier
plaatje.

1. **De geometrie kantelt volledig.** PC2 valt weg van 29% naar 1%: de boog
   wordt een rechte, gelijkmatig verdeelde lijn met correlatie 1,000 tegen de
   getalwaarde. Zodra de lengte van de vector mág meedoen, is een rechte lijn
   de goedkoopste oplossing en gebruikt het model die.

2. **Het lost meteen `9 + 9 = 18` op** — de fout die in experiment 2 bij 9 van
   de 10 seeds terugkwam en die daar nog als "onvermijdelijk gevolg van de
   splitsing" te boek stond. Dat was te somber: 18 komt nog steeds nul keer
   voor in de trainingsdata, maar op een rechte, gelijkmatige lijn is één
   stap voorbij 17 zetten triviaal. Op een boog die terugkromt niet. **De
   extrapolatie was niet onmogelijk, de representatie stond hem in de weg.**
   Ruwe uitvoer voor `9+9`: 17,996.

3. **10 van de 10 seeds foutloos.** Geen enkele fout meer op de
   achtergehouden sommen.

**Let op de reikwijdte.** Dit geldt bij `n_lagen=1`. Layernorm zit er juist
voor diepe stacks, waar de magnitude van `h` bij elke residu-optelling verder
oploopt (zie de docstring van `Blok` in `../experiment/exp.py`). Zet hem terug
aan zodra je lagen toevoegt, en meet dan opnieuw wat er met de geometrie
gebeurt.

**Nieuwe beste configuratie:** `wd=0,3, n_stappen=10000, n_lagen=1,
n_embed=32, gebruik_layernorm=False` — **100% op de achtergehouden sommen,
elke seed.**

---

## 5. Binnenin de attention-laag (`binnenkant.py`)

**Opzet.** Eén laag, vier koppen, vier posities — klein genoeg om helemaal
uit te rekenen. Vraag: wat doen Q, K en V van het `=`-teken, en waar wordt
het optellen eigenlijk gedaan?

**Uitkomst 1 — de aandacht is uniform.** Vanaf de `=`-positie krijgt elke
zichtbare positie exact een kwart, bij alle vier de koppen:

| kop | naar `a` | naar `+` | naar `b` | naar `=` |
|---|---|---|---|---|
| 0 | 0,254 | 0,245 | 0,254 | 0,246 |
| 1–3 | 0,250 | 0,250 | 0,250 | 0,250 |

Spreiding over de 100 sommen: **0,00008**. De affiniteiten vóór de softmax
liggen allemaal tussen −0,031 en +0,011, dus praktisch nul — en `softmax`
van een vlakke rij is een uniforme verdeling. De hele driehoek is 1/1, 1/2,
1/3, 1/4: het model middelt simpelweg alles wat het mag zien.

Dat de aandacht niets doet is direct te toetsen. **Blokkeertest:** vervang de
aandachtsgewichten door precies 1/4 en draai opnieuw. Grootste verandering in
de uitvoer: **0,047**, en alle 100 sommen blijven goed. De verdeling draagt
dus geen informatie.

**Uitkomst 2 — het getal zit in V.** De value-vectoren van `0`..`9`,
geprojecteerd op hun hoofdas:

| | as 1 | correlatie met waarde | stapgrootte |
|---|---|---|---|
| zonder layernorm | 100,00% | **+1,0000** | 0,4905 ± 0,0009 |
| met layernorm | 71% | +0,9374 | niet monotoon |

**Conclusie — zo telt dit netwerk op.** De keten is volledig na te rekenen:

1. De query van `=` is bij élke som **dezelfde vector**: de invoer op die
   positie is `embed('=') + pos(3)`, en die hangt niet van de som af. Er
   valt dus niets te selecteren, en de geleerde Q en K zijn navenant klein
   (`|q|` 0,46 en `|k|` 0,05 op de cijferposities).
2. Uniforme aandacht betekent dat de `=`-positie het **gemiddelde** van de
   vier value-vectoren binnenkrijgt. `v(+)` en `v(=)` zijn constant, dus het
   variabele deel is `(v(a) + v(b)) / 4`.
3. `v(cijfer)` is recht evenredig met de cijferwaarde. Dus
   `(v(a) + v(b)) / 4` is evenredig met **a + b**.

Gemeten: wat de `=`-positie uit attention meekrijgt correleert **+1,00000**
met `a+b` (100,00% van de variantie op één as), en de uiteindelijke uitvoer
correleert +1,000000 met `a+b` met een grootste afwijking van 0,0042.

De optelling gebeurt dus in de **middelingsstap van de aandacht**, niet in
een geleerd opzoekpatroon. Attention wordt hier gebruikt als vaste optelmachine.

**Dit verklaart experiment 4 opnieuw, van binnenuit.** Het mechanisme is bij
beide configuraties identiek — de aandacht is ook mét layernorm uniform
(0,250 ± 0,0003). Wat verschilt is stap 3: mét layernorm is `v(cijfer)` niet
zuiver lineair en aan de onderkant zelfs **niet monotoon** (0, 1, 2 en 3
liggen op één hoop, en 0 ligt hoger dan 2). Daar zat die laatste 5%.

**Vervolgknoppen.** Vier koppen die alle vier hetzelfde doen is verspilling —
werkt `n_koppen=1` net zo goed? En wat doet de feedforward-laag nog, als het
antwoord al uit de attention komt: alleen naschalen, of meer?

---

## 6. Hoe klein kan het? (`binnenkant.py`, ablaties)

**Opzet.** Experiment 5 liet zien dat de aandacht een vaste middelingsstap is
en dat de optelling in V zit. Als dat klopt, zou een hoop van het model
overbodig moeten zijn. Alles over 5 seeds, verder de config van experiment 4.

| variant | test (gem / min) | parameters |
|---|---|---|
| standaard: 4 koppen, met feedforward | 100% / 100% | 13.313 |
| 1 kop | 100% / 100% | 13.313 |
| zonder feedforward | 100% / 100% | 4.961 |
| 1 kop, zonder feedforward | 100% / 100% | 4.961 |
| zonder positie-embedding | 100% / 100% | 13.185 |

Alles tegelijk weglaten (1 kop, geen feedforward, geen positie) en dan
`n_embed` afknijpen:

| n_embed | test (gem / min) | parameters |
|---|---|---|
| 32 | 100% / 100% | 4.833 |
| 8 | 100% / 100% | 441 |
| **4** | **100% / 100%** | **157** |
| 2 | 92% / 85% | 63 |
| 1 | 11% / 0% | 28 |

**Conclusie.**

1. **157 parameters is genoeg** voor foutloos optellen op de achtergehouden
   sommen — 85x kleiner dan waarmee we begonnen. De bodem ligt tussen
   `n_embed` 4 en 2.
2. **Vier koppen doen alle vier hetzelfde**, dus één kop volstaat. Dat sluit
   aan op experiment 5: er valt niets te selecteren, dus valt er ook niets te
   verdelen over koppen.
3. **De feedforward-laag is overbodig.** En dat is geen toeval: de
   lineariteitstoets op het kale model geeft
   `f(a,b) = f(a,0) + f(0,b) - f(0,0)` tot op **0,057** nauwkeurig. Het hele
   netwerk is in feite een lineaire afbeelding — en optellen ís een lineaire
   bewerking, dus er is geen niet-lineariteit nodig. De ReLU had niets te doen.
4. **Positie-embedding is overbodig omdat optellen commutatief is.** Het
   model hoeft `a` en `b` niet uit elkaar te houden; `3 + 7` en `7 + 3` mogen
   op precies dezelfde interne toestand uitkomen.

**Waarschuwing bij het generaliseren hiervan.** Dit zegt iets over déze taak,
niet over transformers. Optellen is lineair en commutatief, en dat is precies
waarom alles wat een transformer onderscheidend maakt — selectieve aandacht,
meerdere koppen, niet-lineariteit, positiegevoeligheid — hier weggelaten kan
worden. Aftrekken (niet commutatief) of vermenigvuldigen (niet lineair) zou
een heel ander antwoord moeten geven. Dat is de scherpste vervolgproef die er
nu ligt.

---

## 7. Aftrekken: wat kost het om niet-commutatief te zijn? (`rekenen.py`, `vergelijk_bewerkingen.py`)

**Opzet.** Experiment 6 eindigde met de voorspelling dat aftrekken een ander
antwoord zou geven, omdat het de commutativiteit mist waar optellen op leunde.
Nieuwe data: `data/aftrekken.txt`, 100 regels `a - b = c` voor a,b van 0 t/m 9,
antwoorden −9 t/m 9. Vocabulaire 21 tokens, net als bij optellen (`-9` is één
token, net zoals `18` dat was). Zelfde splitsing (seed 42, 80/20), zelfde loss.

`rekenen.py` bevat het uitgeklede model uit experiment 6 met de weggehaalde
onderdelen terug als losse knoppen: `positie`, `leer_aandacht` (Q/K, anders
een vaste middeling), `uit_proj` (W_o), `n_koppen` en `ff`.

**Uitkomst 1 — het kale model kán het niet, en dat is te bewijzen.**

| onderdelen | train | test | min | params |
|---|---|---|---|---|
| kaal | 6% | 9% | 5% | 609 |
| + positie | 6% | 10% | 10% | 673 |
| + aandacht (Q/K) | 64% | 60% | **0%** | 1.185 |
| **+ W_o** | **100%** | **100%** | **100%** | **1.457** |
| + 2 of 4 koppen | 100% | 100% | 100% | 1.457 |
| + feedforward | 100% | 100% | 100% | 3.585 |

Bij het kale model en bij `+ positie` geldt `max |f(a,b) − f(b,a)| = 0,000000`
— exact nul. Het model geeft gegarandeerd hetzelfde antwoord op `9-0` en
`0-9` (beide −0,653). Reden: de uitvoer is een gewogen som van de
value-vectoren met gelíjke gewichten, dus symmetrisch in a en b.

**Positie-embeddings helpen daar niets aan**, en dat is de subtielste
uitkomst van dit experiment. Je zou denken dat positie-informatie precies het
ontbrekende stuk is, maar `V` is lineair, dus `V(e + p) = V(e) + V(p)`: de
positie voegt alleen een constante toe en kan het teken van een cijfer niet
omdraaien. Ter controle bleef optellen bij álle varianten op 100%.

**Uitkomst 2 — het is W_o, niet multi-head.** Dat was niet de verwachting.
De redenering vooraf was dat softmax-gewichten niet-negatief zijn, dus dat je
minstens twee koppen nodig hebt om ergens een min voor te krijgen. In de
praktijk volstaat één kop mét `W_o`, en voegen extra koppen niets toe.

**Voorbehoud bij die conclusie.** Wiskundig breidt `W_o` uit wat het model
kan uitdrukken niet uit — het is een lineaire laag tussen twee lineaire
stappen. Het verschil moet dus in de vindbaarheid zitten. Toets: zonder `W_o`
maar 5x zo lang trainen gaat van 60% naar 81% (min 5%) — dat sluit een deel
van het gat, maar niet alles. Trainingsduur alleen verklaart het dus niet, en
deze proef scheidt expressiviteit en vindbaarheid niet schoon.

**Uitkomst 3 — de aandacht gaat écht werken.** Zelfde architectuur, zelfde
config, alleen een andere bewerking:

| | aandacht op `a` | spreiding | correlatie met a |
|---|---|---|---|
| optellen | 0,471 | **0,0003** | +0,27 (ruis) |
| aftrekken | 0,355 | **0,1174** | **+0,887** |

Bij aftrekken kijkt het model naar de **grootste operand**: `9-0` geeft 0,610
op `a`, `0-9` geeft 0,599 op `b`, `5-5` geeft 0,364/0,352. De aandachtskaart
over alle 100 sommen is een gladde diagonale helling — het gewicht hangt af
van `a - b` zelf. Bij optellen is diezelfde kaart volkomen vlak.

**Uitkomst 4 — de prijs.** Kleinste model dat bij élke seed foutloos is:

| bewerking | nodige onderdelen | n_embed | parameters |
|---|---|---|---|
| optellen | geen (kaal) | 2 | **49** |
| aftrekken | positie + aandacht + W_o | 16 | **1.457** |

**Conclusie.** Aftrekken kost een factor 30 aan parameters en dwingt drie
onderdelen af die optellen allemaal kon missen. Belangrijker dan het getal is
wát er verandert: bij optellen is aandacht een vaste middelingsstap die net zo
goed hardgecodeerd kan worden, bij aftrekken is het een inhoudsafhankelijke
selector. Dit is de eerste bewerking in dit spoor waarbij attention doet
waarvoor het bedacht is.

**Eerlijkheidshalve** twee dingen die de vergelijking scheeftrekken, allebei
in het voordeel van aftrekken: bij aftrekken staat elk antwoord ook in de
trainingsset (geen extrapolatie zoals `18` bij optellen), en het bereik van de
antwoorden is kleiner (−9..9 tegen 0..18).

**Nog te doen.** Vermenigvuldigen — de eerste run gaf `kaal` 4% en
`aandacht` 91%, maar die is afgebroken; nog niet betrouwbaar gemeten.

---

## 8. Is het bereik [0,1] van softmax een beperking? (`rekenen.py`, knop `getekend`)

**Opzet.** Vraag naar aanleiding van experiment 7: softmax-gewichten zijn
niet-negatief en tellen op tot 1, dus de aandachtsuitvoer is altijd een
*gewogen gemiddelde* van de value-vectoren — een convexe combinatie, die per
definitie binnen hun omhullende ligt. Een verschil `v(a) − v(b)` ligt
daarbuiten. Voorstel: schaal de gewichten naar [−1, 1] met `2·softmax − 1`.

Wat dat doet is precies uit te schrijven:

```
uitvoer = Σ (2·g_j − 1)·v_j  =  2·Σ g_j v_j  −  Σ v_j
```

Met softmax volledig op `a` worden de gewichten (+1, −1, −1, −1) en is de
uitvoer `v(a) − v(b)` plus constanten. Aftrekken wordt daarmee uitdrukbaar
met een aandachtsverdeling die niet eens van de som hoeft af te hangen.

**Uitkomst**, 10 seeds, `n_embed=16, positie + aandacht`:

| bewerking | gewichten | test | laagste seed |
|---|---|---|---|
| optellen | softmax [0,1] | 100% | **100%** |
| optellen | 2·softmax−1 [−1,1] | 94% | **60%** |
| aftrekken | softmax [0,1] | 78% | **0%** |
| aftrekken | 2·softmax−1 [−1,1] | **100%** | **100%** |

Kleinste foutloze model voor aftrekken:

| oplossing | parameters |
|---|---|
| positie + aandacht + W_o | 1.457 |
| **positie + aandacht + getekend** | **401** |

**Conclusie.**

1. **Ja, het is een echte beperking** — en de vraag legt precies de goede
   vinger op de zere plek. Softmax dwingt een convexe combinatie af, en het
   verschil van twee vectoren ligt daar buiten.

2. **De voorgestelde oplossing werkt, en is 3,6x zuiniger** dan de
   `W_o`-route uit experiment 7: 401 tegen 1.457 parameters. Het gewicht op
   `a` wordt bovendien vrijwel constant (−0,992, spreiding 0,0061, tegen
   0,1174 met softmax) — het model heeft de inhoudsafhankelijke selector niet
   meer nodig omdat een vast ±1-patroon volstaat.

3. **Maar het is een ruil, geen verbetering.** Optellen zakt van 100%/100%
   naar 94%/60%. Precies de eigenschap die dit weghaalt — de uitvoer is een
   gemiddelde — was wat optellen triviaal maakte.

4. **Het schaalt slecht.** De gewichten tellen niet meer op tot 1 maar tot
   `2 − T`. Bij vier posities is dat −2 en onschuldig; bij duizend posities
   −998, en dan stort elke positie standaard −1 in de uitvoer, ongeacht
   relevantie. Dat is waarschijnlijk de reden dat het zo niet gedaan wordt.

5. **Wat echte transformers in plaats hiervan doen** is precies wat we in
   experiment 7 vonden: meerdere koppen die in verschillende stukken van de
   vector schrijven, en `W_o` die die stukken met willekeurige tekens — ook
   negatieve — weer mengt. De convexe combinatie blijft dan binnen elke kop
   intact, en het minteken komt uit de projectie erna. Multi-head attention
   met een uitvoerprojectie bestaat dus mede *omdat* softmax geen negatieve
   gewichten kan maken.

---

## 9. Correctie op experiment 7, en betere manieren om rond nul te schalen

### 9a. Correctie: multi-head helpt aftrekken wél

In experiment 7 staat de conclusie "het is W_o, niet multi-head". **Die klopt
niet.** Ik had daar alleen 2 en 4 koppen getest, met 5 seeds, en te snel
geconcludeerd. Opnieuw, met 10 seeds en verder doorgetest:

| koppen | zonder W_o | met W_o |
|---|---|---|
| 1 | 78% / min 0% | 100% / 100% |
| 2 | 89% / min 0% | 100% / 100% |
| 4 | 98% / min 85% | 100% / 100% |
| **8** | **100% / min 95%** | 100% / 100% |

Multi-head helpt monotoon, en acht koppen halen 100% **zonder enige `W_o`**.
De juiste formulering is dus: `W_o` en multi-head zijn twee routes naar
hetzelfde doel, en `W_o` komt er sneller (één kop volstaat al).

**Het mechanisme is zichtbaar.** Bij 8 koppen zonder `W_o`, aandacht vanaf `=`:

| kop | kijkt naar | helling van de uitlees | rol |
|---|---|---|---|
| 4 | `b` (0,998) | +0,312 | telt b op |
| 6 | `b` (0,987) | +0,197 | telt b op |
| 5 | `a` (0,647) | **−0,367** | telt a af |
| 0,1,3,7 | uniform | ±0,000 | staan uit |

Koppen die naar `a` kijken en koppen die naar `b` kijken krijgen
**tegengestelde tekens** in de uitleeslaag. Dat is precies de constructie die
een verschil mogelijk maakt zonder ooit een negatief softmax-gewicht nodig te
hebben: elke kop houdt zijn eigen convexe combinatie, en het minteken komt uit
de laag erná.

**Voorbehoud.** Die losse bijdragen tellen niet netjes op tot `a − b` — de
optelling geeft ongeveer `−0,24a + 0,51b`. Dat komt doordat de aandacht van
kop 5 meevarieert met de inhoud, waardoor een lineaire ontleding tekortschiet.
De kwalitatieve structuur (gespecialiseerde koppen met tegengestelde tekens)
is aangetoond; een sluitende kwantitatieve reconstructie niet.

### 9b. Vier manieren om rond nul te schalen

Het bezwaar tegen `2·softmax − 1` uit experiment 8 was dat de gewichten
optellen tot `2 − T`. Drie alternatieven, alle met `positie + aandacht`,
n_embed=16, 10 seeds:

| soort | formule | optellen | aftrekken | params | som |
|---|---|---|---|---|---|
| softmax | `g` | 100% / **100%** | 78% / **0%** | 1.185 | +1 |
| getekend | `2g − 1` | 94% / **60%** | 100% / 100% | 1.185 | **2 − T** |
| gecentreerd | `g − gem(g)` | 85% / **25%** | 100% / 100% | 1.185 | 0 |
| verschil | `softmax₁ − softmax₂` | 100% / 95% | 100% / 100% | 1.697 | 0 |
| **tanh** | `tanh(aff)` | **100% / 100%** | **100% / 100%** | **1.185** | vrij |

Gemeten bereik van de gewichten (aftrekken, T=4):

| soort | kleinste | grootste | theoretische grens |
|---|---|---|---|
| gecentreerd | −0,192 | +0,393 | **negatief nooit onder −1/T** |
| verschil | −0,844 | +0,949 | [−1, 1] |
| tanh | −0,966 | +0,954 | [−1, 1] |

**Conclusie.**

1. **`tanh` wint** — 100% op beide bewerkingen bij elke seed, zonder extra
   parameters, en zonder een som die met T meegroeit. De simpelste ingreep
   (normalisatie helemaal weglaten) blijkt de beste.
2. **Gecentreerd lost het T-probleem op maar introduceert een nieuw.** De som
   is netjes 0, maar een negatief gewicht kan nooit onder `−1/T` komen: bij
   T=100 is het sterkste minnetje nog maar −0,01. Dat is te zien aan het
   gemeten bereik, en het verklaart waarom optellen er het hardst onder lijdt
   (85%, min 25%).
3. **Verschil van twee softmaxen werkt goed** (som altijd 0, volle
   tekensterkte) maar kost een tweede stel Q/K: 1.697 tegen 1.185 parameters.
   Merk op dat dit in feite twéé koppen zijn waarvan je de uitvoer aftrekt —
   dus dezelfde oplossing als 9a, in een andere vorm.
4. **Let op wat `tanh` opgeeft.** Zonder normalisatie is er niets dat de
   grootte van de aandachtsuitvoer beteugelt als T groeit; dat is precies
   waarvoor softmax daar zit. Bij T=4 is dat onschuldig. Dit experiment zegt
   dus niets over lange reeksen — daarvoor zou je een taak met variabele
   lengte moeten meten.

**Nagemeten bij het opmaken van de handoff** (10 seeds, aftrekken,
positie + aandacht):

| n_embed | tanh | getekend | parameters |
|---|---|---|---|
| 8 | 100% / 100% | 100% / 100% | 401 |
| **4** | **100% / 100%** | 97% / 80% | **153** |

Aan de kleine kant is `tanh` dus strikt beter, en de bodem voor aftrekken
ligt op **153 parameters** in plaats van de 401 die eerder in dit logboek
stond.

---

## 10. Eén model dat optellen én aftrekken kan (`rekenen.py`, bewerking `"beide"`)

**Opzet.** `data/optellen_aftrekken.txt`: de twee tabellen achter elkaar, 200
regels. Vocabulaire 31 tokens (10 cijfers, 28 antwoorden van −9 t/m 18 met
overlap, plus `+`, `-`, `=`). Splitsing 20% zoals altijd, dus 160 train en 40
test; `N_TEST` is daarvoor vervangen door `TEST_FRACTIE`, wat bij de bestaande
sets van 100 regels precies dezelfde 80/20 oplevert. Geen antwoord komt alleen
in de testset voor, dus geen extrapolatieval.

Dit is wezenlijk een andere taak dan de twee losse: het model moet het
**operator-token lezen** en zijn gedrag daarop omschakelen.

**Uitkomst 1 — het is veel moeilijker dan beide taken apart.** 5 seeds,
n_embed=16:

| onderdelen | train | test | min |
|---|---|---|---|
| kaal | 12% | 9% | 8% |
| + positie | 12% | 9% | 8% |
| + aandacht | 27% | 24% | 8% |
| + W_o | 33% | 29% | 5% |
| + feedforward | 88% | 79% | 40% |
| **+ 4 koppen** | **100%** | **100%** | **100%** |

**Uitkomst 2 — multi-head is hier onmisbaar, anders dan bij de losse taken.**

| koppen | met W_o | zonder W_o |
|---|---|---|
| 1 | 29% | 24% |
| **2** | **100%** | 76% |
| **4** | **100%** | **100%** |
| 8 | 100% | 100% |

Eén kop lukt niet, wat je er ook bij zet. Twee koppen mét `W_o` of vier
koppen zónder volstaan.

**Uitkomst 3 — `tanh` faalt volledig: 8–10% bij élke variant.** Dat is de
gewichtsvorm die in experiment 9 juist beide losse taken foutloos deed. Alle
vormen op de gecombineerde taak (positie + aandacht, n_embed=16):

| soort | test | min |
|---|---|---|
| softmax | 24% | 8% |
| getekend | 45% | 8% |
| gecentreerd | 38% | 5% |
| verschil | 76% | 35% |
| **tanh** | **8%** | **5%** |

**Uitkomst 4 — het mechanisme: aandacht als schakelaar.** Bij 2 koppen +
`W_o`, aandacht vanaf `=`:

| kop | bewerking | naar a | naar teken | naar b |
|---|---|---|---|---|
| 0 | `+` | 0,649 | 0,337 | 0,012 |
| 0 | `−` | 0,766 | 0,217 | 0,015 |
| 1 | `+` | 0,003 | 0,011 | **0,981** |
| 1 | `−` | 0,001 | **0,560** | **0,438** |

Kop 0 leest `a` en verandert nauwelijks. **Kop 1 is de schakelaar**: bij
optellen leest hij `b` op volle sterkte, bij aftrekken verplaatst hij 0,549
van zijn aandacht naar het operator-token — en omdat softmax-gewichten
optellen tot 1, **gaat dat gewicht af van `b`**.

**Conclusie.** Dit verklaart uitkomst 3, en draait de conclusie van experiment
9 gedeeltelijk om. De concurrentie die softmax afdwingt — meer aandacht hier
betekent minder daar — is precies het mechanisme waarmee één token de
berekening omschakelt. Bij `tanh` staan de gewichten los van elkaar en bestaat
die concurrentie niet, dus is er geen schakelaar te bouwen.

**De normalisatie van softmax is dus niet alleen een beperking, maar ook een
gereedschap.** In experiment 8 en 9 zag het eruit als een keurslijf dat
aftrekken onmogelijk maakte; hier is het juist wat conditioneel gedrag
mogelijk maakt. Welk van de twee overheerst hangt af van de taak.

**Kleinste betrouwbare model:**

| bewerking | onderdelen | n_embed | parameters |
|---|---|---|---|
| optellen | geen | 2 | 49 |
| aftrekken | positie + aandacht + tanh | 8 | 401 |
| **beide** | **positie + aandacht + W_o, 4 koppen** | **8** | **553** |

n_embed=4 met 4 koppen komt op 213 parameters en 100% gemiddeld, maar de
laagste seed haalt 98% — net niet betrouwbaar.

---

## 11. Vermenigvuldigen erbij: de feedforward wordt eindelijk nodig (`rekenen.py`, bewerking `"drie"`)

**Opzet.** `data/drie_bewerkingen.txt`: alle drie de tabellen achter elkaar,
300 regels, 53 tokens, antwoorden van −9 t/m 81. Splitsing 20%, dus 240 train
en 60 test, netjes verdeeld (76 keer `+`, 85 keer `-`, 79 keer `*`).
**Bewust met gewone softmax**, niet met `tanh`: dat laatste won bij de losse
bewerkingen maar faalde bij de gecombineerde taak (experiment 10).

Twee antwoorden komen alleen in de testset voor: 21 en 45. Dat zijn allebei
producten (3·7 en 5·9) waarvan beide volgordes in test belandden.

**Uitkomst 1 — de feedforward doet er voor het eerst toe.** 5 seeds:

| onderdelen | n_embed | train | test | min | params |
|---|---|---|---|---|---|
| kaal | 16 | 9% | 5% | 3% | 1.121 |
| positie + aandacht + W_o | 16 | 23% | 25% | 18% | 1.969 |
| + 4 koppen | 16 | 63% | 53% | 12% | 1.969 |
| **+ feedforward** | 16 | 93% | **83%** | 75% | 4.097 |
| + 4 koppen (geen ff) | 32 | 80% | 69% | 53% | 5.985 |
| **+ feedforward** | 32 | 98% | **91%** | 85% | 14.337 |

Bij gelijke grootte scheelt de feedforward dertig procentpunt (53% → 83% bij
n_embed=16, 69% → 91% bij n_embed=32). Dat is de voorspelling uit experiment
6 die uitkomt: optellen en aftrekken zijn lineair en konden de ReLU missen,
vermenigvuldigen is dat niet.

**Uitkomst 2 — langer trainen helpt, groter maken niet.**

| n_embed | koppen | stappen | test | min | params |
|---|---|---|---|---|---|
| 32 | 8 | 10.000 | 94% | 90% | 14.337 |
| **32** | **8** | **30.000** | **98%** | **95%** | **14.337** |
| 64 | 8 | 30.000 | 97% | 95% | 53.249 |
| 64 | 16 | 30.000 | 97% | 95% | 53.249 |

Bijna vier keer zoveel parameters levert niets op; drie keer zo lang trainen
wel.

**Uitkomst 3 — wat er nog misgaat is niet wat je zou denken.** Bij de beste
configuratie, 5 seeds:

| bewerking | goed | aantal in test |
|---|---|---|
| `-` | **100%** | 15 |
| `+` | 99% | 24 |
| `*` | 95% | 21 |

De resterende fouten zijn **allemaal antwoorden vlak bij nul**: `0*9`, `0*8`,
`1*1`, `0+0` — telkens één of twee mis. De twee antwoorden die alleen in de
testset staan (21 en 45) worden juist wél goed geleerd.

**Conclusie.** De bottleneck is niet meer de architectuur maar het
**dynamisch bereik**. Het antwoord moet nu ergens tussen −9 en 81 liggen en op
een halve eenheid nauwkeurig zijn: 0,55% precisie over een spanwijdte van 90.
Bij optellen alleen was dat bereik 18 en dus 2,8% — vijf keer soepeler. Waar
de antwoorden het dichtst opeen liggen, vlak bij nul, glipt het model er het
eerst doorheen. Train-accuratesse is 100%, dus het is geen kwestie van te
weinig capaciteit om te onthouden.

**Beste configuratie:** `n_embed=32, n_koppen=8, positie + aandacht + W_o +
ff, softmax, n_stappen=30000` — 14.337 parameters, **98% (min 95%, max 100%)**.

**Vervolgknoppen.** Het bereikprobleem is te omzeilen door het antwoord anders
te coderen: cijfer voor cijfer voorspellen, of een aparte kop per bewerking.
Allebei veranderen ze de taak, dus dat is een keuze en geen fix.
