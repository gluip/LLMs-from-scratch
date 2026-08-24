"""Proof of concept: emergente woordvectoren i.p.v. karakter- of BPE-tokenisatie.

Zie ontwerp_emergente_woordlaag.html voor de uitleg met plaatjes. Kort: in
plaats van elk karakter apart te embedden (huidig model) of een vaste,
vooraf getrainde BPE-vocabulaire te gebruiken (tiktoken, afgewezen), berekent
dit model een woordvector *uit de letters* van elk woord — geen opzoektabel,
dus het werkt ook voor woorden die nooit in de trainingsdata voorkwamen. Om
diezelfde reden classificeert dit model het volgende woord niet uit een vaste
lijst, maar genereert het karakter voor karakter, gestuurd door de context
van een (ongewijzigd hergebruikte) woord-niveau transformer.

Drie onderdelen, elk met een eigen smoke-test hieronder vóór ze aan elkaar
vastzitten:
  1. KarakterEncoder  - brok (chars) -> 1 woordvector (niet-causaal, mean-pool)
  2. buiten-transformer - reeks woordvectoren -> reeks contextvectoren (causaal,
     kale Blok-stack, RoPE - exact hergebruikt uit exp.py, geen nieuwe klasse)
  3. KarakterDecoder  - contextvector -> volgend woord, letter voor letter

Dit bestand raakt exp.py niet aan (importeert er alleen van), zodat het
stabiele, door 13+ scripts gebruikte exp.py geen risico loopt.
"""
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp import AffiniteitsLaag, Blok, CharTokenizer, APPARAAT, DATA_MAP, TEKST_BESTANDEN, TRAIN_FRACTIE


# ---------------------------------------------------------------------------
# Stap 0: brok-splitsing (spaties = gratis, gegeven grens - niet geleerd)
# ---------------------------------------------------------------------------
def splits_in_brokken(tekst):
    """Splitst in afwisselend niet-witruimte-runs ("woorden", incl. aangehechte
    leestekens) en witruimte-runs. Volledig omkeerbaar: "".join(...) == tekst.
    """
    return re.findall(r"\S+|\s+", tekst)


# ---------------------------------------------------------------------------
# Stap 1: één brok coderen naar vaste lengte, met ruimte voor EOW gereserveerd
# ---------------------------------------------------------------------------
def codeer_brok(brok, tokenizer, max_brok_lengte, eow_id, pad_id):
    """content gecapt op max_brok_lengte-1 tekens, dan altijd precies 1 EOW,
    dan PAD tot de rest - zo past EOW altijd, ook bij een brok die precies zo
    lang is als de cap.
    """
    ids = tokenizer.encode(brok)[: max_brok_lengte - 1]
    ids = ids + [eow_id] + [pad_id] * (max_brok_lengte - 1 - len(ids))
    return ids


# ---------------------------------------------------------------------------
# Stap 2: corpus vooraf omzetten naar een (N, max_brok_lengte)-tensor
# ---------------------------------------------------------------------------
def bouw_brok_tensor(boeken, tokenizer, max_brok_lengte, eow_id, pad_id, train_fractie=TRAIN_FRACTIE):
    """Per boek splitsen+coderen, per boek de laatste `train_fractie`-snede als
    test (zelfde patroon/reden als exp.py: niet de hele testset uit 1 boek).
    """
    train_rijen, test_rijen = [], []
    for _, tekst in boeken:
        brokken = splits_in_brokken(tekst)
        rijen = [codeer_brok(b, tokenizer, max_brok_lengte, eow_id, pad_id) for b in brokken]
        split = int(train_fractie * len(rijen))
        train_rijen += rijen[:split]
        test_rijen += rijen[split:]
    train = torch.tensor(train_rijen, dtype=torch.long)
    test = torch.tensor(test_rijen, dtype=torch.long)
    return train, test


# ---------------------------------------------------------------------------
# Stap 3: batch-trekking op brok-niveau (zelfde truc als maak_batch in exp.py)
# ---------------------------------------------------------------------------
def maak_woord_batch(brok_ids, venster, aantal, generator=None):
    """brok_ids: (N, M). Geeft x_brokken, y_brokken van vorm (aantal, venster, M):
    x = brokken 0..venster-1, y = brokken 1..venster (de doelbrokken die de
    decoder per buiten-positie moet reconstrueren).
    """
    starts = torch.randint(0, len(brok_ids) - venster, (aantal,), generator=generator)
    index = starts.unsqueeze(1) + torch.arange(venster + 1)  # (aantal, venster+1)
    brok = brok_ids[index.to(brok_ids.device)]  # (aantal, venster+1, M)
    return brok[:, :-1], brok[:, 1:]


# ---------------------------------------------------------------------------
# Stap 4: KarakterEncoder - chars van 1 brok -> 1 emergente woordvector
# ---------------------------------------------------------------------------
class KarakterEncoder(nn.Module):
    """Vat de karakters van één brok samen tot één vector - geen opzoektabel,
    dus elke keer herberekend uit de letters. Werkt daardoor ook voor woorden
    die nooit in de trainingsdata voorkwamen.

    Niet-causaal (gebruik_masker=False): alle letters van het woord zijn al
    bekend, dus er is geen reden om vooruitkijken te verbieden zoals bij het
    genereren van nieuwe tekst. Na de Blok-stack volgt masked mean-pooling
    over de niet-PAD-posities (geen CLS-truc: minder machinery, geen extra
    sequentiepositie/RoPE-boekhouding nodig - zie ontwerp_emergente_woordlaag.html).
    PAD-posities tellen wel mee in de aandacht zelf (niet gemaskeerd) - een
    bewuste v1-vereenvoudiging, zie EXPERIMENTEN.md bij dit experiment.
    """

    def __init__(self, char_vocab, pad_id, n_embed_binnen, n_lagen, n_koppen, n_embed_buiten):
        super().__init__()
        # char_vocab = tok.vocab_size (puur de letters); +2 hier voor de PAD- en
        # EOW-rij die de embedding-tabel nodig heeft (die komen ooit als input voor)
        self.pad_id = pad_id
        self.embed = nn.Embedding(char_vocab + 2, n_embed_binnen)
        self.lagen = nn.ModuleList([
            Blok(n_embed_binnen, gebruik_feedforward=True, losse_qk=True, losse_v=True,
                 gebruik_layernorm=True, gebruik_masker=False, n_koppen=n_koppen, uit_projectie=True)
            for _ in range(n_lagen)
        ])
        self.uit_proj = nn.Linear(n_embed_binnen, n_embed_buiten) if n_embed_binnen != n_embed_buiten else None

    def forward(self, brokken):
        # brokken: (Bf, M) karakter-ids -> (Bf, n_embed_buiten) woordvectoren
        h = self.embed(brokken)  # (Bf, M, n_embed_binnen)
        for laag in self.lagen:
            h, _ = laag(h)
        niet_pad = (brokken != self.pad_id).unsqueeze(-1).float()  # (Bf, M, 1)
        som = (h * niet_pad).sum(dim=1)
        aantal = niet_pad.sum(dim=1).clamp(min=1)  # nooit delen door 0 (lege brok kan niet, altijd >=1 char + EOW)
        vector = som / aantal
        return self.uit_proj(vector) if self.uit_proj is not None else vector


# ---------------------------------------------------------------------------
# Stap 6: KarakterDecoder - contextvector -> volgend woord, letter voor letter
# ---------------------------------------------------------------------------
class KarakterDecoder(nn.Module):
    """Typt het volgende woord karakter voor karakter, gestuurd door de
    contextvector uit de buiten-transformer. Vermijdt zo een vaste
    woordenlijst-classificatie: elk denkbaar woord is bereikbaar, ook eentje
    dat nooit in de trainingsdata voorkwam.

    De contextvector wordt als pseudo-positie 0 vóór de karaktersequentie
    geplakt (zelfde M-lengte, karakter c_i op decoderpositie i, dus decoder-
    positie i voorspelt c_{i+1} - exact analoog aan hoe maak_batch in exp.py
    x/y met 1 positie verschuift, alleen zit de "verschuiving" hier al
    ingebakken doordat de context de rol van "voorgaand token" overneemt).
    Causaal (gebruik_masker=True): elk karakter mag alleen terugkijken.

    Deelt de karakter-embedding met KarakterEncoder (scheelt parameters,
    consistent: dezelfde letter heeft overal dezelfde ingangsvector).
    """

    def __init__(self, gedeelde_embed, char_vocab, n_embed_buiten, n_embed_binnen, n_lagen, n_koppen):
        super().__init__()
        self.embed = gedeelde_embed  # nn.Embedding, gedeeld met de encoder
        self.context_proj = nn.Linear(n_embed_buiten, n_embed_binnen)
        self.lagen = nn.ModuleList([
            Blok(n_embed_binnen, gebruik_feedforward=True, losse_qk=True, losse_v=True,
                 gebruik_layernorm=True, gebruik_masker=True, n_koppen=n_koppen, uit_projectie=True)
            for _ in range(n_lagen)
        ])
        self.uit = nn.Linear(n_embed_binnen, char_vocab + 1)  # + EOW, geen PAD-klasse (nooit een geldige voorspelling)

    def forward(self, context, doelbrok):
        # context: (Bf, n_embed_buiten), doelbrok: (Bf, M) -> logits (Bf, M, char_vocab+1)
        Bf, M = doelbrok.shape
        ctx = self.context_proj(context).unsqueeze(1)              # (Bf, 1, n_embed_binnen)
        chars_in = self.embed(doelbrok[:, :-1])                    # (Bf, M-1, n_embed_binnen)
        h = torch.cat([ctx, chars_in], dim=1)                      # (Bf, M, n_embed_binnen)
        for laag in self.lagen:
            h, _ = laag(h)
        return self.uit(h)


# ---------------------------------------------------------------------------
# Stap 5+7: HierarchischModel - encoder + buiten-transformer + decoder
# ---------------------------------------------------------------------------
class HierarchischModel(nn.Module):
    """Combineert de drie onderdelen. Let op: dit is NIET AffiniteitsModel
    hergebruikt (die heeft zelf een nn.Embedding/nn.Linear-naar-vocab die we
    juist willen vermijden) - de buiten-transformer is een kale Blok-lijst.
    """

    def __init__(self, tok_vocab_size, pad_id, eow_id, max_brok_lengte,
                 n_embed_binnen, n_lagen_enc, n_lagen_dec, n_koppen_binnen,
                 n_embed_buiten, n_lagen_buiten, n_koppen_buiten, brok_venster, dropout=0.0):
        super().__init__()
        self.pad_id = pad_id
        self.eow_id = eow_id
        self.max_brok_lengte = max_brok_lengte
        self.encoder = KarakterEncoder(tok_vocab_size, pad_id, n_embed_binnen,
                                        n_lagen_enc, n_koppen_binnen, n_embed_buiten)
        self.buiten_lagen = nn.ModuleList([
            Blok(n_embed_buiten, gebruik_feedforward=True, losse_qk=True, losse_v=True,
                 gebruik_layernorm=True, gebruik_masker=True, dropout=dropout, n_koppen=n_koppen_buiten,
                 uit_projectie=True, gebruik_rope=True, max_lengte=brok_venster)
            for _ in range(n_lagen_buiten)
        ])
        self.decoder = KarakterDecoder(self.encoder.embed, tok_vocab_size, n_embed_buiten,
                                        n_embed_binnen, n_lagen_dec, n_koppen_binnen)

    def forward(self, x_brokken, y_brokken):
        # x_brokken/y_brokken: (aantal, venster, M)
        aantal, venster, M = x_brokken.shape
        platte_x = x_brokken.reshape(aantal * venster, M)
        # De encoder is een pure functie van de karakters (dezelfde tekst ->
        # altijd dezelfde vector), dus twee keer "de" of een spatie hoeft maar
        # 1x verwerkt te worden. In de praktijk is ~70% van de brokken in een
        # batch een letterlijke herhaling (spaties en veelvoorkomende woordjes
        # domineren) - dedupliceren voor de encoder scheelt dus fors rekenwerk,
        # zonder de uitkomst te veranderen (torch.unique+gather is wiskundig
        # identiek aan alles los verwerken). Geldt niet voor de decoder: die
        # krijgt per positie een andere contextvector mee, dus twee keer "de"
        # op verschillende plekken heeft daar wél verschillende invoer.
        unieke_x, terug_index = torch.unique(platte_x, dim=0, return_inverse=True)
        unieke_vectoren = self.encoder(unieke_x)
        woordvectoren = unieke_vectoren[terug_index].view(aantal, venster, -1)  # (aantal, venster, n_embed_buiten)
        h = woordvectoren
        for laag in self.buiten_lagen:
            h, _ = laag(h)  # (aantal, venster, n_embed_buiten)
        platte_context = h.reshape(aantal * venster, -1)
        platte_y = y_brokken.reshape(aantal * venster, M)
        logits = self.decoder(platte_context, platte_y)  # (aantal*venster, M, char_vocab+1)
        return logits


# ---------------------------------------------------------------------------
# Stap 8: training - zelfde AdamW+cosine-schema als train_affiniteitsmodel
# ---------------------------------------------------------------------------
def train_hierarchisch(model, train_brokken, test_brokken, brok_venster, aantal,
                        lr, n_stappen, eval_interval, apparaat, seed=0):
    torch.manual_seed(seed)
    model = model.to(apparaat)
    train_brokken = train_brokken.to(apparaat)
    test_brokken = test_brokken.to(apparaat)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_stappen)
    n_par = sum(p.numel() for p in model.parameters())
    char_vocab_out = model.decoder.uit.out_features  # tok.vocab_size + 1

    print(f"trainen ({n_par:,} parameters, {n_stappen} stappen, venster={brok_venster} brokken) op {apparaat}...")
    for stap in range(n_stappen):
        x_b, y_b = maak_woord_batch(train_brokken, brok_venster, aantal)
        logits = model(x_b, y_b)
        loss = F.cross_entropy(logits.reshape(-1, char_vocab_out), y_b.reshape(-1), ignore_index=model.pad_id)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

        if stap % eval_interval == 0 or stap == n_stappen - 1:
            model.eval()
            with torch.no_grad():
                x_t, y_t = maak_woord_batch(test_brokken, brok_venster, aantal)
                logits_t = model(x_t, y_t)
                loss_t = F.cross_entropy(logits_t.reshape(-1, char_vocab_out), y_t.reshape(-1), ignore_index=model.pad_id)
            model.train()
            huidige_lr = optimizer.param_groups[0]["lr"]
            print(f"  stap {stap:>5}  train loss {loss.item():.3f}  test loss {loss_t.item():.3f}  lr {huidige_lr:.5f}")
    return model


# ---------------------------------------------------------------------------
# Stap 9: genereren - geneste lus (buiten: woord-voor-woord, binnen: karakter-voor-karakter)
# ---------------------------------------------------------------------------
def genereer_hierarchisch(model, tokenizer, start_brokken, brok_venster, n_nieuwe_woorden,
                           apparaat, temperatuur=0.8, generator=None):
    """start_brokken: lijst van reeds-gecodeerde brokken (elk M lang) om mee te
    beginnen. Genereert n_nieuwe_woorden extra brokken en geeft de volledige
    tekst terug (start + nieuw), plus de gegenereerde woorden los (voor de
    lengte/bestaand-woord-validatie).
    """
    model.eval()
    M = model.max_brok_lengte
    alle_ids = list(start_brokken)  # lijst van (M,)-lijsten
    nieuwe_woorden_tekst = []

    with torch.no_grad():
        for _ in range(n_nieuwe_woorden):
            venster_ids = alle_ids[-brok_venster:]
            x = torch.tensor([venster_ids], dtype=torch.long, device=apparaat)  # (1, T, M)
            T = x.shape[1]
            woordvectoren = model.encoder(x.reshape(T, M)).view(1, T, -1)
            h = woordvectoren
            for laag in model.buiten_lagen:
                h, _ = laag(h)
            context = h[:, -1]  # (1, n_embed_buiten) - context na het laatste brok

            # binnenlus: karakter voor karakter, tot EOW of M-1
            gegenereerd = []
            huidig = torch.full((1, M), model.pad_id, dtype=torch.long, device=apparaat)
            for i in range(M - 1):
                logits = model.decoder(context, huidig)  # (1, M, char_vocab+1)
                kansen = torch.softmax(logits[0, i] / temperatuur, dim=-1)
                if generator is not None:
                    volgend = torch.multinomial(kansen.cpu(), num_samples=1, generator=generator).item()
                else:
                    volgend = torch.multinomial(kansen, num_samples=1).item()
                if volgend == model.eow_id:
                    break
                gegenereerd.append(volgend)
                # doelbrok[k] hoort op h-positie k+1 (zie KarakterDecoder.forward:
                # chars_in = embed(doelbrok[:, :-1])) - dus het net gegenereerde
                # karakter voor positie i terug op INDEX i zetten, niet i+1, anders
                # schuift de causale context een plek op en genereert de decoder
                # blind (dit was een echte bug, zie EXPERIMENTEN.md)
                huidig[0, i] = volgend
            woord_ids = gegenereerd + [model.eow_id] + [model.pad_id] * (M - 1 - len(gegenereerd))
            alle_ids.append(woord_ids)
            nieuwe_woorden_tekst.append(tokenizer.decode(gegenereerd))

    volledige_tekst = "".join(tokenizer.decode([i for i in brok if i not in (model.pad_id, model.eow_id)])
                               for brok in alle_ids)
    return volledige_tekst, nieuwe_woorden_tekst


if __name__ == "__main__":
    # --- smoke test stap 0: round-trip ---
    boeken = [(n, (DATA_MAP / n).read_text(encoding="utf-8")) for n in TEKST_BESTANDEN[:3]]
    for naam, tekst in boeken:
        brokken = splits_in_brokken(tekst)
        assert "".join(brokken) == tekst, f"round-trip kapot bij {naam}"
    print(f"stap 0 ok: round-trip klopt op {len(boeken)} boeken")

    # even een gevoel bij de brok-lengtes (voor MAX_BROK_LENGTE straks)
    tok = CharTokenizer("".join(t for _, t in boeken))
    alle_brokken = splits_in_brokken("".join(t for _, t in boeken))
    lengtes = sorted(len(b) for b in alle_brokken)
    n = len(lengtes)
    print(f"aantal brokken: {n:,}  mediaan lengte: {lengtes[n//2]}  "
          f"90e percentiel: {lengtes[int(n*0.9)]}  99e percentiel: {lengtes[int(n*0.99)]}  "
          f"max: {lengtes[-1]}")

    # --- smoke test stap 1: codeer_brok op handmatige voorbeelden ---
    # EOW moet een geldige uitvoerklasse van de decoder zijn (die classificeert
    # over char_vocab+1 = tok.vocab_size+1 klassen, indices 0..tok.vocab_size),
    # dus EOW komt eerst. PAD wordt nooit voorspeld (alleen als input gebruikt,
    # via ignore_index uit de loss gehouden) en mag daarom wél buiten dat bereik
    # liggen.
    EOW_ID = tok.vocab_size
    PAD_ID = tok.vocab_size + 1
    M = 8
    voorbeelden = ["kat", "kattenkwaad", " ", "\n\n", "!"]
    for v in voorbeelden:
        ids = codeer_brok(v, tok, M, EOW_ID, PAD_ID)
        assert len(ids) == M
        assert EOW_ID in ids, f"geen EOW in {v!r} -> {ids}"
        terug = tok.decode([i for i in ids if i not in (PAD_ID, EOW_ID)])
        verwacht = v[: M - 1]
        assert terug == verwacht, f"{v!r} -> decode {terug!r} != verwacht {verwacht!r}"
        print(f"  {v!r:>16} -> {ids}")
    print(f"stap 1 ok: codeer_brok klopt op {len(voorbeelden)} voorbeelden (M={M}, PAD={PAD_ID}, EOW={EOW_ID})")

    # --- smoke test stap 2+3: corpus omzetten + batch trekken ---
    M_SMOKE = 12
    train_brokken, test_brokken = bouw_brok_tensor(boeken, tok, M_SMOKE, EOW_ID, PAD_ID)
    print(f"\nstap 2 ok: train_brokken {tuple(train_brokken.shape)}  test_brokken {tuple(test_brokken.shape)}")

    g = torch.Generator().manual_seed(0)
    x_b, y_b = maak_woord_batch(train_brokken, venster=8, aantal=4, generator=g)
    assert x_b.shape == (4, 8, M_SMOKE) and y_b.shape == (4, 8, M_SMOKE)
    # y op positie i moet exact x op positie i+1 zijn (1 brok opgeschoven)
    assert torch.equal(x_b[:, 1:], y_b[:, :-1])
    print(f"stap 3 ok: x_b {tuple(x_b.shape)}  y_b {tuple(y_b.shape)}, verschuiving klopt")
    print("  voorbeeld brok (eerste van batch, positie 0):",
          [tok.int_naar_char.get(i, "PAD" if i == PAD_ID else "EOW") for i in x_b[0, 0].tolist()])

    # --- smoke test stap 4: KarakterEncoder determinisme ---
    torch.manual_seed(0)
    encoder = KarakterEncoder(char_vocab=tok.vocab_size, pad_id=PAD_ID,
                               n_embed_binnen=24, n_lagen=1, n_koppen=2, n_embed_buiten=48)
    encoder.eval()
    woorden = ["kat", "kat", "hond", "katten"]
    ids = torch.tensor([codeer_brok(w, tok, M_SMOKE, EOW_ID, PAD_ID) for w in woorden])
    with torch.no_grad():
        vecs = encoder(ids)
    assert torch.allclose(vecs[0], vecs[1]), "zelfde woord ('kat' x2) gaf verschillende vectoren!"
    assert not torch.allclose(vecs[0], vecs[2]), "verschillende woorden ('kat'/'hond') gaven dezelfde vector!"
    assert not torch.allclose(vecs[0], vecs[3]), "'kat'/'katten' gaven identieke vector (verdacht)"
    print(f"\nstap 4 ok: KarakterEncoder deterministisch (vorm {tuple(vecs.shape)}), "
          f"'kat'=='kat' maar 'kat'!='hond' en 'kat'!='katten'")

    # --- smoke test stap 6: KarakterDecoder geisoleerd laten overfitten ---
    # 8 herhaalde identieke (willekeurige) contextvectoren -> 8x hetzelfde doelwoord;
    # als de architectuur leerbaar is, moet de loss naar ~0 zakken
    torch.manual_seed(1)
    N_EMBED_BINNEN, N_EMBED_BUITEN = 24, 48
    decoder = KarakterDecoder(encoder.embed, char_vocab=tok.vocab_size,
                               n_embed_buiten=N_EMBED_BUITEN, n_embed_binnen=N_EMBED_BINNEN,
                               n_lagen=1, n_koppen=2)
    doelwoord = torch.tensor([codeer_brok("vrolijk", tok, M_SMOKE, EOW_ID, PAD_ID)] * 8)
    vaste_context = torch.randn(8, N_EMBED_BUITEN)
    opt = torch.optim.Adam(decoder.parameters(), lr=3e-3)
    for stap in range(150):
        logits = decoder(vaste_context, doelwoord)
        loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size + 1), doelwoord.reshape(-1), ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step()
        if stap % 50 == 0 or stap == 149:
            print(f"  decoder-overfit stap {stap:>3}  loss {loss.item():.4f}")
    assert loss.item() < 0.05, f"decoder overfit niet naar ~0 (eindloss {loss.item():.4f}) - architectuur-bug?"
    print("stap 6 ok: KarakterDecoder is leerbaar (overfit op 8 herhaalde brokken lukt)")

    # =========================================================================
    # Volle smoke-run: het hele model, klein/goedkoop, op 1 boek
    # =========================================================================
    print("\n" + "=" * 70)
    print("SMOKE-RUN: heel HierarchischModel, klein, op pinkeltje.txt")
    BROK_VENSTER = 8
    boek_pinkeltje = [(n, t) for n, t in boeken if n == "pinkeltje.txt"]
    train_pk, test_pk = bouw_brok_tensor(boek_pinkeltje, tok, M_SMOKE, EOW_ID, PAD_ID)
    print(f"pinkeltje.txt: {train_pk.shape[0]:,} train-brokken, {test_pk.shape[0]:,} test-brokken")

    torch.manual_seed(0)
    model = HierarchischModel(
        tok_vocab_size=tok.vocab_size, pad_id=PAD_ID, eow_id=EOW_ID, max_brok_lengte=M_SMOKE,
        n_embed_binnen=24, n_lagen_enc=1, n_lagen_dec=1, n_koppen_binnen=2,
        n_embed_buiten=48, n_lagen_buiten=2, n_koppen_buiten=2, brok_venster=BROK_VENSTER,
    )
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par:,} parameters")

    t0 = time.time()
    N_STAPPEN_SMOKE = 300
    model = train_hierarchisch(model, train_pk, test_pk, brok_venster=BROK_VENSTER, aantal=8,
                                lr=5e-3, n_stappen=N_STAPPEN_SMOKE, eval_interval=50, apparaat=APPARAAT)
    duur = time.time() - t0
    print(f"\nsmoke-run klaar: {duur:.1f}s voor {N_STAPPEN_SMOKE} stappen "
          f"({1000*duur/N_STAPPEN_SMOKE:.1f}ms/stap) op {APPARAAT}")

    # --- validatie 1: buurwoorden (cosinus-gelijkenis) ---
    print("\n--- validatie: buurwoorden ---")
    model.eval()
    paren = [("huis", "huizen"), ("huis", "visser"), ("speelde", "speelden"), ("de", "het")]
    woorden_flat = sorted(set(w for p in paren for w in p))
    with torch.no_grad():
        ids = torch.tensor([codeer_brok(w, tok, M_SMOKE, EOW_ID, PAD_ID) for w in woorden_flat], device=APPARAAT)
        vecs = model.encoder(ids)
    vec_van = dict(zip(woorden_flat, vecs))
    for a, b in paren:
        sim = F.cosine_similarity(vec_van[a].unsqueeze(0), vec_van[b].unsqueeze(0)).item()
        print(f"  cos({a!r}, {b!r}) = {sim:+.3f}")

    # --- validatie 2+3+4: genereren, dan lengteverdeling + percentage bestaande woorden ---
    print("\n--- validatie: genereren ---")
    start_tekst = "Pinkeltje "
    start_brokken = [codeer_brok(b, tok, M_SMOKE, EOW_ID, PAD_ID) for b in splits_in_brokken(start_tekst)]
    volledige_tekst, nieuwe_woorden = genereer_hierarchisch(
        model, tok, start_brokken, brok_venster=BROK_VENSTER, n_nieuwe_woorden=20,
        apparaat=APPARAAT, temperatuur=0.8)
    print(f"  gegenereerd: {volledige_tekst!r}")

    echte_brokken_pk = splits_in_brokken(boek_pinkeltje[0][1])
    echte_woorden = set(b for b in echte_brokken_pk if b.strip())
    echte_lengtes = sorted(len(b) for b in echte_woorden)
    gen_woorden = [w for w in nieuwe_woorden if w.strip()]
    gen_lengtes = sorted(len(w) for w in gen_woorden) if gen_woorden else [0]
    n_echt = len(echte_lengtes)
    n_gen = len(gen_lengtes)
    print(f"  woordlengte mediaan: echt={echte_lengtes[n_echt//2]}  gegenereerd={gen_lengtes[n_gen//2]}")
    aantal_bestaand = sum(1 for w in gen_woorden if w in echte_woorden)
    print(f"  percentage bestaande woorden: {aantal_bestaand}/{len(gen_woorden)} "
          f"({100*aantal_bestaand/max(1,len(gen_woorden)):.0f}%)  (na maar {N_STAPPEN_SMOKE} stappen op 1 boek - laag is normaal)")

    print("\nsmoke-run volledig geslaagd - alle onderdelen werken en zijn leerbaar.")
