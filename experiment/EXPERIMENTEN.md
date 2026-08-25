# Experimentenlogboek

Bijhouden wat er al geprobeerd is, met uitkomst en conclusie — zodat we niet
over een paar weken blind hetzelfde experiment nog eens draaien. **Voeg na elk
nieuw vergelijkingsscript een entry toe** (onderaan, chronologisch), en werk de
tabel hieronder bij als er een nieuw beste model komt.

Alle getallen zijn "loss op de laatste positie" uit `loss_per_positie()` in
`exp.py`, tenzij anders vermeld — dat is de betrouwbaarste maat die we
gebruiken (gemiddeld over 20 batches van 256, specifiek op de positie met de
volle context). De "test"-loss die tijdens training zelf print is ruwer (één
losse batch, gemiddeld over het hele venster inclusief de dure vroege
posities) en kan een andere kant op wijzen dan de laatste-positie-loss — zie
bijvoorbeeld experiment 10 en 12 hieronder. Vertrouw bij twijfel op de
laatste-positie-loss.

**Ruis:** identieke configuraties geven van run tot run ~0,01–0,02 verschil
(GPU/cuBLAS-nondeterminisme, ook bij vaste seed). Een verschil kleiner dan
dat is geen echt signaal.

## Vaste instellingen

Deze stonden de hele sessie ongewijzigd op de standaardwaarde uit `exp.py`
(module-constanten `N_LAGEN`, `N_KOPPEN`, etc.), bij álle experimenten
hieronder, tenzij een entry expliciet iets anders zegt. Bij elk experiment
staat daarom alleen wat er *voor dat experiment specifiek* toe doet — deze
lijst hoeft niet herhaald te worden:

`n_lagen=5, n_koppen=4, uit_projectie=True, gebruik_layernorm=True,
gebruik_feedforward=True, gebruik_masker=True, ff_factor=4.0,
aantal_train=64, aantal_test=256, train_fractie=0.9, seed=0`

## Huidige beste configuratie (`model.pt`)

| instelling | waarde |
|---|---|
| n_embed | 160 |
| n_lagen | 5, n_koppen 4 |
| losse_qk / losse_v | True / True (volledig Q/K/V) |
| gebruik_rope | True (`gebruik_positie=False`) |
| dropout | 0,1 |
| lr | 5e-3 |
| lengte (training) | 64 |
| n_stappen | 18000 |
| dataset | 14 boeken, 10,2M karakters (`TEKST_BESTANDEN` in `exp.py`) |
| **laatste-positie-loss** | **1,2605** |

Back-ups van eerdere "beste modellen": `model_zonder_rope.pt` (zelfde config
zonder RoPE, absolute posities, 1,2781), `model_3boeken.pt` (zelfde config op
de oorspronkelijke 3 boeken i.p.v. 14, 1,2455).

## Huidige beste hiërarchische configuratie (`model_hierarchisch.pt`)

Aparte architectuur (chars -> emergente woordvector -> transformer -> chars,
zie `ontwerp_emergente_woordlaag.html`), dus een apart model naast `model.pt` -
niet hetzelfde bestand, andere tokenizer-benadering, ander praat-script
(`praat_hierarchisch.cmd`).

| instelling | waarde |
|---|---|
| n_embed_buiten | 160, **5 lagen**, 4 koppen, RoPE (breedte/diepte hier getest, geen bottleneck - experiment 20/21) |
| n_embed_binnen (encoder/decoder) | 128, **6 lagen** elk, 4 koppen (diepte was de belangrijkste hefboom - experiment 21/22) |
| MAX_BROK_LENGTE / BROK_VENSTER | 16 / **48** (~144 karakters - experiment 23/24) |
| dropout / lr | 0,1 / 3e-3 (5e-3 gaf instabiliteit bij 128/4, zie experiment 17/18) |
| n_stappen | 18000 |
| dataset | zelfde 14 boeken, 10,2M karakters |
| **nats/karakter (inhoud, EOW uitgesloten)** | **1,1288** — **wint van `model.pt` (1,2605)** met 0,132 |

`HierarchischModel.forward` dedupliceert sinds experiment 18 de encoder-
aanroep (`torch.unique(..., dim=0)`) - ~70% van de brokken in een batch is
een letterlijke herhaling (spaties, "de", "een", ...), dus dat scheelt 27%
rekentijd per stap zonder de uitkomst te veranderen. `exp.py`'s `Blok` kreeg
sinds experiment 21 een `ff_factor`-parameter (default ongewijzigd).

Drie assen zijn uitputtend gesweept (breedte, diepte, venstergrootte, elk
aan binnen- en/of buitenkant) - zie experiment 20-24 voor de volledige
zoektocht en de samenvattende tabel in experiment 24. Nog open: dropout op
de binnenkant, de niet-gemaskeerde PAD-posities in de encoder-attention, en
`MAX_BROK_LENGTE` zelf (nooit los getest).

Back-ups van tussentijdse versies: `model_hierarchisch_v4_128-4.pt` (128/4
binnen, venster 32, 1,1730), `model_hierarchisch_v5_binnen6.pt` (128/6
binnen, venster 32, 1,1527 - vóór de venster-uitbreiding naar 48).

## Geprobeerd en verworpen — niet zomaar opnieuw proberen

- **QKV breder dan n_embed** (`qkv_factor` 1x/1,5x/2x, ~zomer 2026). Geen
  verbetering, verschil viel binnen de ruis (1,2781/1,2747/1,2728). Volledig
  teruggedraaid, zit niet meer in `exp.py`. Zie experiment 11.
- **22-boeken-dataset met pre-1947-spelling teksten** (Max Havelaar, Sara
  Burgerhart, Ideën I-III, Camera Obscura, Huis Lauernesse, De Roos van
  Dekama). Spelling ("hy", "zy", "vryer") lekte door in gegenereerde tekst.
  Niet opnieuw toevoegen zonder de spelling eerst te moderniseren. Zie
  experiment 8.
- **Langer trainen dan 18000 stappen** op de huidige datasetgrootte (getest
  tot 36000). Geen zinnig verschil — de data is dan al meerdere keren
  herhaald en de test-loss ligt al vanaf ~stap 10000 plat. Zie experiment 7.

## Log

### 1. Venstergrootte (context length): 16 / 32 / 64 / 128
- **Script:** `venster_vgl.py`
- **Parameters:** `n_embed=80, lengte=[16,32,64,128], dropout=0.0, lr=1e-2,
  n_stappen=5000, losse_qk=True, losse_v=True, gebruik_positie=True,
  dataset=3 boeken (1,8M tekens, 1.623.458 train)`
- **Opzet:** oorspronkelijke 3 boeken (1,8M tekens), n_embed=80, lr=1e-2.
- **Uitkomst:** 16→1,459, 32→1,403, 64→1,363, 128→1,358 (laatste positie).
  64→128 levert nog maar 0,005 op.
- **Conclusie:** `LENGTE=64` is de zoete plek; verder vergroten kost
  rekentijd voor verwaarloosbare winst op deze datasetgrootte.

### 2. n_embed (80 vs 160) x venstergrootte
- **Script:** `embed_vgl.py`
- **Parameters:** `n_embed=[80,160], lengte=[32,64,128], dropout=0.0, lr=1e-2,
  n_stappen=5000, losse_qk=True, losse_v=True, gebruik_positie=True,
  dataset=3 boeken (1,8M tekens)`
- **Opzet:** lr vast op 1e-2 (nog niet gesweept), 5000 stappen.
- **Uitkomst:** n_embed=160 presteerde slechter dan 80 op elk venster,
  vooral bij 128 (1,551). Train-loss van 160 was ook hoger dan van 80 →
  onderfitting, geen overfitting.
- **Conclusie:** lr=1e-2 is te grof voor een model van 1,6M parameters →
  aanleiding voor experiment 3.

### 3. Leerrate x n_embed x venster (op 5000 stappen — later herzien!)
- **Script:** `lr_vgl.py`
- **Parameters:** `n_embed=[80,160], lengte=[64,128], dropout=0.0,
  lr=[3e-3,5e-3,1e-2 (1e-2 hergebruikt uit experiment 2)], n_stappen=5000,
  losse_qk=True, losse_v=True, gebruik_positie=True, dataset=3 boeken (1,8M tekens)`
- **Opzet:** lr ∈ {3e-3, 5e-3, 1e-2}, n_embed ∈ {80, 160}, venster ∈ {64, 128}.
- **Uitkomst:** n_embed=160, lr=5e-3, venster=64 leek de winnaar (1,315).
- **⚠️ Waarschuwing:** dit was op maar 5000 stappen. Bij de volle 18000
  stappen bleek dezelfde configuratie juist zwaar te overfitten (zie
  experiment 4) — **hyperparameters kiezen op een verkorte run is
  misleidend zodra modelgrootte een rol speelt**, want overfitting hangt af
  van trainingsduur.

### 4. Dezelfde configuraties op de volle 18000 stappen
- **Script:** `finale_vgl.py`
- **Parameters:** `lengte=64, dropout=0.0, n_stappen=18000, losse_qk=True,
  losse_v=True, gebruik_positie=True, dataset=3 boeken (1,8M tekens)`;
  varianten: standaard `n_embed=80, lr=1e-2` vs sweep-winnaar
  `n_embed=160, lr=5e-3`
- **Opzet:** standaard (n_embed=80, lr=1e-2) tegen de "winnaar" uit
  experiment 3 (n_embed=160, lr=5e-3), nu op 18000 stappen.
- **Uitkomst:** standaard: test 1,3262, train 1,1645, gat +0,16. Sweep-winnaar:
  test 1,5874, train 0,887, gat **+0,70** (zware overfitting).
- **Conclusie:** de 5000-stappen-conclusie klopte niet. Extra capaciteit
  (n_embed=160) zonder regularisatie overfit zwaar bij lange training →
  aanleiding voor de dropout-sweep.

### 5. Dropout-sweep voor n_embed=160
- **Script:** `dropout_vgl.py`
- **Parameters:** `n_embed=160, lengte=64, lr=5e-3, n_stappen=18000,
  dropout=[0.0 (hergebruikt uit experiment 4), 0.1, 0.2, 0.3], losse_qk=True,
  losse_v=True, gebruik_positie=True, dataset=3 boeken (1,8M tekens)`
- **Opzet:** n_embed=160, lr=5e-3, 18000 stappen, dropout ∈ {0,0, 0,1, 0,2, 0,3}.
- **Uitkomst:** 0,0→1,5874 (gat +0,70), **0,1→1,2455 (gat +0,05)**,
  0,2→1,2492, 0,3→1,2833.
- **Conclusie:** dropout=0,1 lost de overfitting van experiment 4 op én
  verslaat de oude standaard (1,2455 vs 1,3262) — grootste losse winst van de
  hele hyperparameter-zoektocht. Dit werd de nieuwe standaardconfiguratie.

### 6. Langer trainen: 18000 vs 36000 stappen
- **Scripts:** `beste_loss.py`, `langer_trainen.py`
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=[18000,36000], losse_qk=True, losse_v=True, gebruik_positie=True,
  dataset=3 boeken (1,8M tekens)`
- **Opzet:** winnende configuratie uit experiment 5, op de (toen nog)
  3-boeken-dataset (1,8M tekens).
- **Uitkomst:** 18k stappen: test 1,3175. 36k stappen: test 1,3033 — vrijwel
  gelijk, curve ligt al vanaf ~stap 10000 plat.
- **Conclusie:** bij 18000 stappen wordt de trainingsset al ~45x herhaald —
  de data was op, niet de trainingstijd. Meer stappen op dezelfde data helpt
  niet meer → aanleiding om de dataset uit te breiden (experiment 8).

### 7. Dataset-uitbreiding, eerste poging: 22 boeken (verworpen)
- **Script:** `train_uitgebreid.py` (eerste run)
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=18000, losse_qk=True, losse_v=True, gebruik_positie=True,
  dataset=22 boeken (18,7M tekens, 16.866.000 train, vocab=202)`
- **Opzet:** 19 DBNL-titels toegevoegd aan de oorspronkelijke 3, incl. Max
  Havelaar, Sara Burgerhart, Ideën I-III, Camera Obscura, Huis Lauernesse, De
  Roos van Dekama → 18,7M tekens.
- **Uitkomst:** test-loss ging omhóóg naar 1,3761 (was 1,2455). Gegenereerde
  tekst bevatte pre-1947-spelling ("Hy had hem niet geweest", "vryer wil").
- **Conclusie:** die zes titels zijn overwegend in pre-1947-spelling en
  vervuilden het Nederlands van het hele model. Verwijderd (zie "Geprobeerd
  en verworpen" hierboven).

### 8. Dataset-uitbreiding, schoongemaakt: 14 boeken
- **Script:** `train_uitgebreid.py` (na opschoning van `TEKST_BESTANDEN`)
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=18000, losse_qk=True, losse_v=True, gebruik_positie=True,
  dataset=14 boeken (10,2M tekens, 9.209.727 train, vocab=148)`
- **Opzet:** de zes probleemtitels uit experiment 7 verwijderd; 11 overige
  DBNL-titels behouden (allemaal <2,5% pre-1947-spellingmarkers) → 10,2M
  tekens, 5,7x de oorspronkelijke dataset.
- **Uitkomst:** test-loss 1,2781 — bijna gelijk aan de zuivere 3-boeken-versie
  (1,2455), maar op 5,7x zoveel data en zonder spellingvervuiling.
- **Conclusie:** dit werd de dataset-standaard. Zie ook de opmerking in de
  code: overgebleven boeken zijn nog wel 19e/vroeg-20e-eeuws qua woordkeuze
  ("gij", "den") — dat is stijl, geen spellingfout, en inherent aan
  publiek-domein Nederlandse literatuur.

### 9. Eén gedeelde matrix vs losse Q/K/V
- **Script:** `qkv_vgl.py`
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=18000, gebruik_positie=True, dataset=14 boeken (10,2M tekens)`;
  varianten: `losse_qk=False, losse_v=False` vs `losse_qk=True, losse_v=True`
- **Opzet:** `losse_qk=False, losse_v=False` (1 matrix W) tegen
  `losse_qk=True, losse_v=True` (volledig Q/K/V), verder de standaard-config.
- **Uitkomst:** 1 matrix: 1,3612 (1.345.748 parameters). Losse Q/K/V:
  **1,2781** (1.601.748 parameters, +19%).
- **Conclusie:** losse Q/K/V wint duidelijk en structureel (de hele
  leercurve ligt lager, niet pas aan het eind) — bevestigt waarom dit al de
  standaard was. Zie ook de theoretische onderbouwing in de docstring van
  `AffiniteitsLaag`: één matrix laat elke positie vooral op zichzelf letten.

### 10. QKV breder dan n_embed (`qkv_factor`) — verworpen
- **Script:** `qkv_breder_vgl.py` (verwijderd na dit experiment, niet meer
  in de repo)
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=18000, losse_qk=True, losse_v=True, uit_projectie=True (verplicht
  zodra qkv_factor≠1), gebruik_positie=True, dataset=14 boeken (10,2M tekens)`;
  variant: `qkv_factor=[1.0, 1.5, 2.0]`
- **Opzet:** nieuwe knop `qkv_factor` in `AffiniteitsLaag` om Q/K/V breder te
  maken dan n_embed (1x/1,5x/2x), los van de rest van het model.
- **Uitkomst:** 1x: 1,2781 (test 1,3217, sanity-check ok). 1,5x: 1,2747
  (test 1,3236). 2x: 1,2728 (test 1,3243).
- **Conclusie:** verschil tussen varianten (0,0053) valt binnen de ruis
  (~0,01–0,02) — geen echt effect. `qkv_factor` volledig teruggedraaid uit
  `exp.py` op verzoek. Attention-capaciteit was bij 40 dim/kop al voldoende;
  breder maken helpt niet, in tegenstelling tot experiment 9 waar de
  *structuur* van attention veranderde (niet alleen de breedte).

### 11. Absolute positie-embeddings vs RoPE
- **Script:** `rope_vgl.py`
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=18000, losse_qk=True, losse_v=True, uit_projectie=True,
  dataset=14 boeken (10,2M tekens)`; varianten:
  `gebruik_positie=True, gebruik_rope=False` vs
  `gebruik_positie=False, gebruik_rope=True`
- **Opzet:** `gebruik_positie=True, gebruik_rope=False` (huidige standaard)
  tegen `gebruik_positie=False, gebruik_rope=True`, verder identiek.
- **Uitkomst:** zonder RoPE: test 1,3217, laatste positie 1,2781
  (1.601.748 parameters). Met RoPE: test 1,3673, laatste positie **1,2605**
  (1.591.508 parameters — geen `pos_embed`-tabel meer nodig).
- **Conclusie:** RoPE wint op de betrouwbare maat (laatste positie) met
  minder parameters, al is de winst klein (0,0176, net boven de ruis). Werd
  het nieuwe `model.pt` (zie experiment 13). Let op: "test"-loss wijst hier
  de andere kant op — zie de uitleg over de twee maten bovenaan dit bestand.

### 12. RoPE-extrapolatie: generaliseert het naar een langer venster?
- **Script:** `rope_extrapolatie.py`
- **Parameters:** zelfde als experiment 11 (`n_embed=160, dropout=0.1,
  lr=5e-3, n_stappen=18000, losse_qk=True, losse_v=True, uit_projectie=True,
  dataset=14 boeken`), getraind op `lengte=64`, geëvalueerd op
  `lengte=[64, 128]` zonder hertraining (`LENGTE_LANG=128`)
- **Opzet:** beide modellen uit experiment 11 getraind op lengte=64, daarna
  zónder hertraining geëvalueerd op lengte=128. Voor RoPE: de hoektabel
  (`rope_cos`/`rope_sin`, pure functie van positie) doorgerekend tot 128.
  Voor absolute posities: `pos_embed` aangevuld met verse, ongetrainde rijen
  voor positie 64–127 (zoals er ook zou gebeuren zonder de afkap-bescherming
  in `genereer()`).
- **Uitkomst:** op lengte=128: zonder RoPE 2,9986 (+1,73 t.o.v. lengte=64,
  een klif direct na positie 64). Met RoPE 2,5681 (+1,30, een geleidelijke
  helling, geen klif).
- **Conclusie:** RoPE haalt het harde architecturale plafond weg (geen
  ongetrainde tabelrijen meer), maar lost het generalisatieprobleem niet
  automatisch op — de geleerde Q/K-gewichten zijn nooit geoptimaliseerd voor
  relatieve afstanden >63, dus er blijft degradatie. Om daar iets aan te
  doen zou je op een langer venster moeten trainen (nog niet geprobeerd).

### 13. RoPE-model opgeslagen als `model.pt`
- **Script:** `train_rope.py`
- **Parameters:** `n_embed=160, lengte=64, dropout=0.1, lr=5e-3,
  n_stappen=18000, losse_qk=True, losse_v=True, uit_projectie=True,
  gebruik_positie=False, gebruik_rope=True, dataset=14 boeken (10,2M tekens)`
- **Opzet:** de winnende configuratie uit experiment 11 nogmaals getraind
  (18000 stappen, 14-boeken-dataset) en opgeslagen.
- **Uitkomst:** laatste-positie-loss 1,2605 — reproduceerde experiment 11
  exact. Oude model (absolute posities) veiliggesteld als
  `model_zonder_rope.pt`.

## Nog niet geprobeerd (ideeën uit de architectuur-brainstorm)

Bij het zoeken naar "relaties die het model moeilijk kan leggen" kwamen twee
concrete, onderbouwde kandidaten naar boven (zie ook de rest van dit repo:
`ch04/08_deltanet`, `ch05/07_gpt_to_llama` e.a. voor bestaande
implementaties elders in het boek):

- **RoPE + trainen op een langer venster** (bv. lengte=128 i.p.v. 64), om de
  klif/helling uit experiment 12 daadwerkelijk dicht te trainen in plaats
  van 'm alleen architecturaal mogelijk te maken.
- **DeltaNet-achtig associatief geheugen** — een vast-formaat
  geheugentoestand die per stap bijwerkt via een Hebbiaanse delta-regel
  (geïnspireerd door Hopfield-netwerken/statistische mechanica uit de
  neurowetenschap), voor relaties die sowieso buiten één trainingsvenster
  vallen — iets wat RoPE per definitie niet oplost, want dat blijft werken
  binnen één forward pass.
- Verder, minder uitgewerkt: automatentheorie/stack-geheugen voor geneste
  structuur (haakjes, aanhalingstekens), numerieke cognitie/plaatswaarde-
  encodering voor cijferrelaties, fonetische features voor rijm.

### 14. Emergente woordvectoren: chars -> geleerde woordvector -> transformer -> chars

- **Scripts:** `hierarchisch.py` (bouwstenen + smoke-tests), `train_hierarchisch.py`
  (volle run), zie `ontwerp_emergente_woordlaag.html` voor de uitleg met plaatjes.
- **Parameters:** `MAX_BROK_LENGTE=16, BROK_VENSTER=32 (~64 karakters),
  n_embed_binnen=64, n_lagen_enc=2, n_lagen_dec=2, n_koppen_binnen=4,
  n_embed_buiten=160, n_lagen_buiten=5, n_koppen_buiten=4, dropout=0.1, lr=5e-3,
  aantal=64, n_stappen=18000 (35,6ms/stap gemeten, ~10,5 min), gebruik_rope=True
  op de buitenste stack, dataset=14 boeken (10,2M tekens)`
- **Opzet:** i.p.v. tiktoken/BPE (afgewezen: vaste, op Engels getrainde
  vocabulaire) een architectuur zonder énige vaste woordenlijst. Drie
  onderdelen: (1) `KarakterEncoder` - berekent per brok (woord of witruimte-run)
  één vector uit de letters, niet-causaal, masked mean-pooling, géén opzoektabel
  dus werkt ook op nooit-geziene woorden; (2) kale `Blok`-stack (ongewijzigd
  hergebruikt uit exp.py, causaal, RoPE) over de reeks woordvectoren; (3)
  `KarakterDecoder` - genereert het volgende woord letter voor letter uit de
  context, i.p.v. classificatie uit een vaste lijst. PAD/EOW-boekhouding voor
  vaste brok-lengte (content gecapt op M-1, dan EOW, dan PAD).
- **Uitkomst:** eerlijke vergelijking (nats/karakter, EOW uitgesloten) tegen de
  char-baseline (1,2605): **1,2963** (+0,0358) - opvallend dicht bij de
  baseline voor een fundamenteel andere architectuur die zijn eigen
  woordrepresentaties vanaf nul moest leren in dezelfde trainingstijd.
  Buurwoorden-check toont bescheiden maar echte structuur: `cos(huis,huizen)
  =0,871`, `cos(speelde,speelden)=0,947`, `cos(mooi,mooie)=0,955` allemaal
  hoger dan `cos(de,het)=0,837` en `cos(Pinkeltje,kabouter)=0,693`.
- **⚠️ Bug gevonden via de validatie-checks, niet via de loss:** de eerste
  volle run gaf onleesbare tekst (`"m m e o m seern..."`, 2% bestaande
  woorden) ondanks een normaal ogend loss-getal. Oorzaak: `genereer_hierarchisch`
  zette een net gegenereerd karakter op index `i+1` in plaats van `i` in de
  decoder-invoer (`KarakterDecoder.forward` plaatst `doelbrok[k]` op
  h-positie `k+1`), waardoor de causale context tijdens genereren een plek
  opschoof. Trainen zelf gebruikte de kant-en-klare `doelbrok`-tensor en had
  deze bug niet - vandaar het misleidend normale loss-getal naast kapotte
  tekst. Na de fix (zelfde getrainde gewichten, geen hertraining nodig):
  **100% bestaande woorden bij temperatuur 0,6, 92% bij 0,8, 78% bij 1,0**,
  en leesbaar, grammaticaal Nederlands ("Op een dag bij de kamer te stoorde
  en zich in alle trekken te bescheiden. Zij had den kamer voor de
  gelegenheid gevonden...").
- **Conclusie:** dit is precies het scenario waar de validatie-checks voor
  gebouwd zijn (zie de "hoe checken we of de woorden OK zijn"-sectie in
  `ontwerp_emergente_woordlaag.html`) - de loss-cijfers alleen hadden deze
  bug nooit blootgelegd. Met de fix is dit een sterke, werkende proof of
  concept: kwalitatief vergelijkbaar met het char-model, zonder ooit een
  vaste woordenlijst te gebruiken. **Nog niet gepromoveerd tot `model.pt`**
  (apart opgeslagen als `model_hierarchisch.pt`, `praat.py` weet er nog niks
  van) - dat is een bewuste volgende stap, geen automatisme, zoals bij elk
  eerder experiment deze sessie.

### 15. Char-model tegen hiërarchisch model, direct naast elkaar

- **Script:** `hierarchisch_vs_char_vgl.py` (laadt beide al-getrainde modellen,
  geen hertraining), `praat_hierarchisch.py`/`.cmd` (los REPL-script, zusje
  van `praat.cmd`, zodat je ze interactief naast elkaar kunt proberen).
- **Uitkomst:** char-model 1,2605 nats/char (1.591.508 parameters) tegen
  hiërarchisch 1,3004-1,3050 nats/char (1.783.157 parameters, 12% meer) - het
  char-model won op dat moment, buiten de meetruis. Op dezelfde prompts bleef
  het hiërarchische model herkenbaar Nederlands maar iets schokkeriger, met
  af en toe een verzonnen woord.
- **Conclusie:** geen verrassing - het char-model heeft 13 experimenten
  hypertuning achter zich, het hiërarchische model nog geen enkele (leende
  lr/dropout klakkeloos van het char-model). Aanleiding voor experiment 16.

### 16. Sweep van het hiërarchische model — en de winnaar

- **Script:** `hierarchisch_sweep.py`
- **Parameters (vast):** `MAX_BROK_LENGTE=16, BROK_VENSTER=32, n_embed_buiten=160,
  n_lagen_buiten=5, n_koppen_buiten=4, n_koppen_binnen=4, aantal=64,
  n_stappen=18000 (volle runs, geen verkorte-run-fout zoals experiment 3),
  dataset=14 boeken`; varianten: baseline (lr5e-3,drop0.1,n_embed_binnen=64,
  2 lagen), dropout=0, lr=8e-3, en n_embed_binnen=96 met 3 lagen encoder/decoder.
- **Diagnose vooraf:** de trainingscurve van de eerste poging had een klein
  train/test-gat (~0,1-0,15) - geen overfitting-signatuur zoals bij het
  char-model zonder dropout, eerder een teken dat het model nog niet tegen
  een plafond aanliep. Dat wees op capaciteit, niet op leerrate/regularisatie.
- **Uitkomst:** baseline 1,3050, dropout=0 1,3047, lr=8e-3 1,3058 - alle drie
  identiek binnen de ruis, bevestigt dat lr/dropout niet de bottleneck waren.
  **Grotere binnen-encoder/decoder (96 dim, 3 lagen): 1,2139** - een
  duidelijke, structurele verbetering (zichtbaar vanaf ~stap 1500 in de
  loss-curve, niet pas aan het eind) en **wint van de char-baseline (1,2605)**
  met 0,047 nats, ondanks nog steeds geen enkele vaste woordenlijst.
- **Conclusie:** de bottleneck van de eerste poging zat inderdaad in
  representatiecapaciteit van de karakter-encoder/-decoder, niet in
  optimalisatie. Dit is nu de nieuwe standaard-hiërarchische-configuratie,
  opgeslagen als `model_hierarchisch.pt` (oude versie: `model_hierarchisch_v1.pt`).
  Nog niet gesweept: `MAX_BROK_LENGTE`/`BROK_VENSTER`, nog grotere binnen-
  capaciteit (is 96/3 lagen zelf al een plafond, of kan het verder groeien?),
  en `n_embed_buiten` los van het char-model optimaliseren in plaats van
  klakkeloos matchen.

### 17. Nog groter, of langer trainen? (vervolg op experiment 16)

- **Script:** `hierarchisch_sweep2.py`
- **Parameters (vast):** zelfde als experiment 16, dataset=14 boeken. De
  18k/96-3-winnaar zelf is hergebruikt uit `hierarchisch_sweep_resultaten.pt`,
  niet opnieuw getraind.
- **Varianten:** `n_embed_binnen=128, n_lagen_enc/dec=4, n_stappen=18000` (nog
  groter) tegen `n_embed_binnen=96, n_lagen_enc/dec=3, n_stappen=36000`
  (langer trainen op de bekende winnaar).
- **Uitkomst "nog groter" (128/4): 1,4752 — duidelijk slechter**, niet beter.
  Oorzaak zichtbaar in de trainingscurve: een echte instabiliteit (train-loss
  piekte naar 10,2 rond stap 7500, gradient-explosie, geen ruis) waar het
  model nooit meer volledig van herstelde. Dit is *geen* bewijs dat 96/3 een
  hard capaciteitsplafond is - het laat zien dat `lr=5e-3` (klakkeloos
  overgenomen van de 96/3-config) te grof is voor een groter/dieper model,
  hetzelfde patroon als bij het char-model in experiment 2/3 (lr die voor de
  ene omvang werkt, faalt bij een grotere zonder aanpassing).
- **Uitkomst "langer trainen" (96/3, 36k): 1,1882 — beter dan de 18k-winnaar
  (1,2139)**, stabiele curve, geen instabiliteit. Bevestigt de diagnose: het
  model was bij 18000 stappen nog niet uitgeleerd, in tegenstelling tot het
  char-model waar `langer_trainen.py` liet zien dat de data toen al
  verzadigd was (experiment 6) - dit model doet per stap een moeilijkere,
  samengestelde taak en heeft kennelijk meer herhaling nodig.
- **Conclusie:** langer trainen was de winnende hefboom, niet groter maken.
  Nieuwe standaard-hiërarchische-configuratie: 96/3, 36000 stappen, **1,1882
  nats/char — 0,072 beter dan de char-baseline (1,2605)**. Opgeslagen als
  `model_hierarchisch.pt` (18k-versie veiliggesteld als
  `model_hierarchisch_v2_18k.pt`). Open vraag voor een volgende sessie: zou
  128/4 (of groter) wél winnen bij een lagere leerrate of warmup, gegeven dat
  de instabiliteit en niet de capaciteit zelf de tegenvaller verklaarde?

### 18. Nog verder: 72k stappen, en 128/4 met een lagere leerrate

- **Script:** `hierarchisch_sweep3.py`
- **Parameters (vast):** zelfde als experiment 16/17, dataset=14 boeken.
- **Varianten:** `n_embed_binnen=96, 3 lagen, n_stappen=72000` (nog langer,
  na het succes van 18k->36k) en `n_embed_binnen=128, 4 lagen, n_stappen=18000,
  lr=3e-3` (de instabiele 128/4 uit experiment 17 opnieuw, nu met een lagere
  leerrate dan de 5e-3 die daar een gradient-explosie gaf).
- **Uitkomst 128/4, lr=3e-3: 1,1730 — stabiel** (hoogste train-loss na
  opwarmen: 1,22, geen piek zoals bij lr=5e-3) **en beter dan de 36k/96-3-
  winnaar (1,1882), in de helft van de stappen (18k tegen 36k).** Bevestigt
  de hypothese uit experiment 17 hard: de instabiliteit was een leerrate-
  probleem, geen capaciteitsplafond. Nieuwe standaard-hiërarchische-
  configuratie, opgeslagen als `model_hierarchisch.pt` (36k-versie
  veiliggesteld als `model_hierarchisch_v3_36k.pt`).
- **Zijspoor - een echte snelheidsoptimalisatie gevonden:** tijdens het
  wachten kwam de vraag of de batch-dimensie wel goed gebruikt werd. Dat
  bleek te kloppen, maar leverde wel een concrete optimalisatie op: gemeten
  op één trainingsbatch was van de 2048 "woord-slots" (aantal x venster)
  maar 610 uniek - **70,2% van het werk van de karakter-encoder was
  letterlijke herhaling** (spaties alleen al 1009x in één batch, plus
  veelvoorkomende woordjes als "de"/"een"/"en"/"van"). De encoder is een
  pure functie van de karakters, dus dat is puur verspild werk. Fix in
  `HierarchischModel.forward`: `torch.unique(..., dim=0, return_inverse=True)`
  vóór de encoder-aanroep, resultaat terug uitgesmeerd met de inverse-index.
  Geldt niet voor de decoder (die krijgt per positie een andere contextvector
  mee, dus geen herhaling om te benutten). Correctheid geverifieerd op CPU:
  logits exact identiek (verschil 0,0), gradiënten identiek tot op
  afrondingsniveau (~1,9e-6). Gemeten snelheidswinst: 48ms/stap tegen 65,5ms/
  stap voorheen - **27% sneller**, zonder enige verandering in de uitkomst.
- **Conclusie:** twee onafhankelijke winsten uit één ronde - een betere
  hyperparameter-configuratie (128/4, lr=3e-3) én een gratis snelheids-
  optimalisatie (encoder-deduplicatie) die alle toekomstige runs met dit
  model versnelt, ook als de hyperparameters weer veranderen.

### 19. 72k stappen op 96/3 - bevestigt een écht capaciteitsplafond

- **Script:** `hierarchisch_sweep3.py` (herstart ná de dedup-fix, dus met de
  27% snellere encoder - de 72k stappen kostten 59,1 min i.p.v. de ~80 min
  die zonder de fix nodig waren geweest)
- **Uitkomst: 1,1748** - vrijwel gelijk aan de 36k-uitkomst uit experiment 17
  (1,1882) en de 18k/128-4-winnaar (1,1730), allemaal binnen de meetruis.
  In `hierarchisch_sweep3.png` is goed te zien waarom: de curve van 96/3 ligt
  al vanaf ~stap 10.000-15.000 plat en blijft daar de resterende 57.000
  stappen hangen - geen kwestie van trainingsduur meer op dat moment, maar
  een echt capaciteitsplafond van die modelgrootte. De 128/4-curve zakt in
  minder stappen naar een lager niveau en blijft daar.
- **Conclusie:** "langer trainen" was de juiste hefboom tussen 18k en 36k
  (experiment 17), maar is bij 96/3 nu uitgewerkt - verder trainen op déze
  grootte helpt niet meer. De winnende configuratie blijft **128/4, lr=3e-3,
  18000 stappen: 1,1730 nats/char**, al gepromoveerd tot `model_hierarchisch.pt`.
  Open vraag voor een volgende sessie: ligt bij 128/4 hetzelfde plafond-
  patroon te wachten bij meer stappen, of geeft de grotere capaciteit ook
  daar meer ruimte om te blijven verbeteren?

### 20. n_embed_buiten los getest (was klakkeloos overgenomen van het char-model)

- **Script:** `hierarchisch_sweep4.py`
- **Parameters (vast):** winnende binnenconfig (n_embed_binnen=128, 4 lagen
  enc/dec, lr=3e-3, dropout=0,1, n_stappen=18000, dataset=14 boeken);
  variant: `n_embed_buiten` ∈ {160 (huidig, hergebruikt), 224, 288}.
- **Aanleiding:** de binnenkant liet afnemende meeropbrengst zien (64/2->96/3:
  +0,091 nats; 96/3->128/4: nog maar +0,041), terwijl `n_embed_buiten=160`
  sinds het begin gewoon overgenomen was van het char-model, nooit los
  getest voor deze architectuur.
- **Uitkomst:** 224 -> 1,1810, 288 -> 1,1805 - **beide net iets slechter dan
  160 (1,1730)**, en de twee liggen vrijwel op elkaar ondanks het verschil
  in grootte. In `hierarchisch_sweep4.png` liggen de curves de hele training
  door structureel boven de 160-lijn, niet pas toevallig aan het eind.
- **Conclusie:** de buitenste transformer is hier niet de bottleneck -
  groter maken helpt niet, in tegenstelling tot de binnenkant. `n_embed_buiten
  =160` blijft staan. Dit is verder ook een compliment aan de oorspronkelijke,
  destijds ongeteste keuze: hij bleek toevallig al goed, al was dat nooit
  geverifieerd voordat dit experiment het bevestigde.

### 21. Lagen (buiten/binnen) en ff_factor los getest - diepte is de hefboom

- **Script:** `hierarchisch_sweep5.py`. Vereiste een kleine, backward-
  compatible toevoeging aan `exp.py`: `Blok` kreeg een nieuwe `ff_factor`-
  parameter (default `FF_FACTOR`, dus geen enkele bestaande aanroep
  verandert) die puur een doorgeefluik is naar `FeedForwardLaag`. Smoke-
  getest: bestaand pad ongewijzigd, nieuwe knop werkt (`factor=2.0` ->
  binnen-dim 64 bij n_embed=32, zoals verwacht).
- **Parameters (vast):** winnende config (n_embed_binnen=128, n_embed_buiten
  =160, lr=3e-3, dropout=0,1, n_stappen=18000). Per variant wijzigt 1 ding
  t.o.v. de winnaar (1,1730):

| variant | nats/char | vs. winnaar |
|---|---|---|
| buiten 3 lagen (was 5) | 1,1710 | -0,002 |
| buiten 8 lagen (was 5) | 1,1720 | -0,001 |
| binnen 2 lagen (was 4, zelfde breedte 128) | 1,2153 | **+0,042** |
| **binnen 6 lagen (was 4, zelfde breedte 128)** | **1,1527** | **-0,020** |
| ff_factor binnen 2.0 (was 4.0) | 1,1944 | +0,021 |
| ff_factor binnen 8.0 (was 4.0) | 1,1712 | -0,002 |

- **Uitkomst:** de buitenste transformer is ongevoelig voor zowel breedte
  (experiment 20) als diepte (3, 5 en 8 lagen liggen allemaal binnen 0,002
  van elkaar) - geen bottleneck, op geen van beide assen. `ff_factor` breder
  dan 4.0 helpt niet, smaller (2.0) kost duidelijk kwaliteit. **Alleen de
  diepte van de binnenste encoder/decoder is nog een echte hefboom**: van 4
  naar 2 lagen kost 0,042 (fors), van 4 naar 6 lagen wint nog eens 0,020.
  Dit isoleert wat de eerdere 64/2->96/3->128/4-progressie deed: niet (alleen)
  de breedte, vooral de diepte.
- **Conclusie:** **nieuwe winnaar - n_embed_binnen=128, 6 lagen (was 4),
  verder ongewijzigd: 1,1527 nats/char**, 0,108 beter dan de char-baseline
  (1,2605). Gepromoveerd tot `model_hierarchisch.pt`. Duidelijke vervolgvraag:
  blijft dieper nog verder winnen (8 lagen binnen), of is dit ook een
  plafond zoals bij de buitenkant?

### 22. 8 lagen binnen, en een goedkopere combinatie geprobeerd

- **Script:** `hierarchisch_sweep6.py`
- **Varianten:** `n_lagen_enc/dec=8` (nog dieper dan de 6-lagen-winnaar) en
  een gecombineerde variant (binnen 6 lagen + buiten 3 lagen, in de
  veronderstelling dat de buitenkant toch ongevoelig was voor diepte -
  experiment 21 - dus goedkoper zou moeten kunnen zonder kwaliteit te
  verliezen).
- **Uitkomst:** 8 lagen binnen: **1,1551** - vrijwel gelijk aan de 6-lagen-
  winnaar (1,1527), verschil binnen de meetruis. Diepte is dus ook hier
  uitgewerkt: 6 lagen is het optimum op deze as, net zoals eerder bij
  n_embed_buiten en n_lagen_buiten een plafond bleek. Gecombineerd (6
  binnen/3 buiten): **1,1682 - 0,016 slechter** dan 6/5. De vereenvoudiging
  pakt dus niet gunstig uit: hoewel buiten-diepte alléén (bij binnen=4)
  geen verschil maakte, kost het blijkbaar wél iets zodra de binnenkant
  krachtiger is (6 lagen) - een interactie-effect dat losse tests per as
  niet laten zien.
- **Conclusie:** de winnende configuratie blijft **n_embed_binnen=128, 6
  lagen encoder/decoder, n_embed_buiten=160, 5 lagen buiten: 1,1527
  nats/char**. Twee assen zijn nu uitputtend verkend (breedte en diepte, aan
  beide kanten) en leveren geen verdere winst meer op zonder een andere
  aanpak (bv. `MAX_BROK_LENGTE`/`BROK_VENSTER`, dropout op de binnenkant, of
  de niet-gemaskeerde PAD-posities in de encoder-attention - alle drie nog
  open staande punten uit eerdere experimenten).

### 23. BROK_VENSTER los getest - meer context helpt, net als bij het char-model

- **Script:** `hierarchisch_sweep7.py`. Winnende architectuur (128/6 binnen,
  160/5 buiten, lr=3e-3) vast; alleen `BROK_VENSTER` varieert (16/32/48).
  `MAX_BROK_LENGTE=16` ongewijzigd, dus dezelfde brok-tensor hergebruikt.
- **Aanleiding:** `BROK_VENSTER=32` was vanaf het begin gekozen om qua
  *karakters* te matchen met het char-model se `LENGTE=64`, nooit los
  getest - en bij het char-model was de venstergrootte destijds de
  waardevolste knop van allemaal.
- **Uitkomst:** `BROK_VENSTER=16`: **1,2327 - fors slechter** (+0,080).
  `BROK_VENSTER=48`: **1,1288 - beter** (-0,024). Een duidelijke,
  monotone trend: meer woord-context helpt, net als bij het char-model.
  In `hierarchisch_sweep7.png` ligt de 48-curve de hele training door
  structureel onder zowel de 16- als de 32-lijn, niet toevallig pas aan
  het eind.
- **Conclusie:** **nieuwe winnaar - BROK_VENSTER=48 (was 32), verder
  ongewijzigd: 1,1288 nats/char**, 0,132 beter dan de char-baseline
  (1,2605). Gepromoveerd tot `model_hierarchisch.pt`. Gegeven de duidelijke
  trend (16 fors slechter, 48 beter) is de logische vervolgvraag: helpt
  nóg meer venster (64+) verder, of is dit ook een plafond?

### 24. BROK_VENSTER=64 - de trend vlakt af

- **Script:** `hierarchisch_sweep8.py`, vervolg op experiment 23.
- **Uitkomst:** **1,1316** - vrijwel gelijk aan de 48-winnaar (1,1288),
  verschil binnen de meetruis. De reeks 16(1,2327)->32(1,1527)->48(1,1288)
  ->64(1,1316) laat duidelijk afnemende en dan omslaande winst zien: -0,080,
  dan -0,024, dan +0,003. `BROK_VENSTER=48` is het optimum op deze as.
- **Conclusie:** geen nieuwe winnaar, `BROK_VENSTER=48` blijft de
  standaard. Drie assen zijn nu uitputtend verkend (breedte, diepte,
  venstergrootte) en hebben allemaal hun plafond laten zien. Overzicht van
  de hele zoektocht (char-baseline 1,2605):

| stap | wijziging | nats/char |
|---|---|---|
| start (64/2, ongesweept) | - | 1,3050 |
| binnenkant breder+dieper | 64/2 -> 96/3 -> 128/4 | 1,2139 -> 1,1730 |
| leerrate gefixt | 5e-3 -> 3e-3 (128/4 werd instabiel op 5e-3) | 1,1730 |
| binnenkant dieper (breedte gelijk) | 4 -> 6 lagen | 1,1527 |
| venster groter | 32 -> 48 brokken | **1,1288** |

  Nog open (niet meer opgepakt deze sessie, tijd op): dropout op de
  binnenste encoder/decoder, de niet-gemaskeerde PAD-posities in de
  encoder-attention, en `MAX_BROK_LENGTE` (nu 16, nooit los getest - alleen
  `BROK_VENSTER` is dat wel geweest).
