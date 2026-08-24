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
- **Opzet:** oorspronkelijke 3 boeken (1,8M tekens), n_embed=80, lr=1e-2.
- **Uitkomst:** 16→1,459, 32→1,403, 64→1,363, 128→1,358 (laatste positie).
  64→128 levert nog maar 0,005 op.
- **Conclusie:** `LENGTE=64` is de zoete plek; verder vergroten kost
  rekentijd voor verwaarloosbare winst op deze datasetgrootte.

### 2. n_embed (80 vs 160) x venstergrootte
- **Script:** `embed_vgl.py`
- **Opzet:** lr vast op 1e-2 (nog niet gesweept), 5000 stappen.
- **Uitkomst:** n_embed=160 presteerde slechter dan 80 op elk venster,
  vooral bij 128 (1,551). Train-loss van 160 was ook hoger dan van 80 →
  onderfitting, geen overfitting.
- **Conclusie:** lr=1e-2 is te grof voor een model van 1,6M parameters →
  aanleiding voor experiment 3.

### 3. Leerrate x n_embed x venster (op 5000 stappen — later herzien!)
- **Script:** `lr_vgl.py`
- **Opzet:** lr ∈ {3e-3, 5e-3, 1e-2}, n_embed ∈ {80, 160}, venster ∈ {64, 128}.
- **Uitkomst:** n_embed=160, lr=5e-3, venster=64 leek de winnaar (1,315).
- **⚠️ Waarschuwing:** dit was op maar 5000 stappen. Bij de volle 18000
  stappen bleek dezelfde configuratie juist zwaar te overfitten (zie
  experiment 4) — **hyperparameters kiezen op een verkorte run is
  misleidend zodra modelgrootte een rol speelt**, want overfitting hangt af
  van trainingsduur.

### 4. Dezelfde configuraties op de volle 18000 stappen
- **Script:** `finale_vgl.py`
- **Opzet:** standaard (n_embed=80, lr=1e-2) tegen de "winnaar" uit
  experiment 3 (n_embed=160, lr=5e-3), nu op 18000 stappen.
- **Uitkomst:** standaard: test 1,3262, train 1,1645, gat +0,16. Sweep-winnaar:
  test 1,5874, train 0,887, gat **+0,70** (zware overfitting).
- **Conclusie:** de 5000-stappen-conclusie klopte niet. Extra capaciteit
  (n_embed=160) zonder regularisatie overfit zwaar bij lange training →
  aanleiding voor de dropout-sweep.

### 5. Dropout-sweep voor n_embed=160
- **Script:** `dropout_vgl.py`
- **Opzet:** n_embed=160, lr=5e-3, 18000 stappen, dropout ∈ {0,0, 0,1, 0,2, 0,3}.
- **Uitkomst:** 0,0→1,5874 (gat +0,70), **0,1→1,2455 (gat +0,05)**,
  0,2→1,2492, 0,3→1,2833.
- **Conclusie:** dropout=0,1 lost de overfitting van experiment 4 op én
  verslaat de oude standaard (1,2455 vs 1,3262) — grootste losse winst van de
  hele hyperparameter-zoektocht. Dit werd de nieuwe standaardconfiguratie.

### 6. Langer trainen: 18000 vs 36000 stappen
- **Scripts:** `beste_loss.py`, `langer_trainen.py`
- **Opzet:** winnende configuratie uit experiment 5, op de (toen nog)
  3-boeken-dataset (1,8M tekens).
- **Uitkomst:** 18k stappen: test 1,3175. 36k stappen: test 1,3033 — vrijwel
  gelijk, curve ligt al vanaf ~stap 10000 plat.
- **Conclusie:** bij 18000 stappen wordt de trainingsset al ~45x herhaald —
  de data was op, niet de trainingstijd. Meer stappen op dezelfde data helpt
  niet meer → aanleiding om de dataset uit te breiden (experiment 8).

### 7. Dataset-uitbreiding, eerste poging: 22 boeken (verworpen)
- **Script:** `train_uitgebreid.py` (eerste run)
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
