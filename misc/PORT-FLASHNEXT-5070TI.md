# Porting Flash-Next su ds4 per RTX 5070 Ti + 62 GB RAM

### ULTIMO AGGIORNAMENTO (leggi questo per primo)

**1. BUG DI CORRETTEZZA trovato e corretto: kernel batch IQ2_XXS.**
Mancava l'offset `ib32` sul puntatore delle attivazioni (`ys[b][i].qs` invece di
`ys[b][i].qs + 32*ib32`): ogni iterazione consuma 16 byte di pesi = 64 valori, quindi
il kernel rileggeva i primi 64 valori per tutte e 4 le iterazioni del blocco,
ignorando 3/4 delle attivazioni.  **Errore misurato: 1,11e+01 (1110%)**.
IQ2_XXS copre **9 dei 48 layer MoE del modello ISTA** (quello in uso): quei layer
producevano contributi MoE sbagliati in ogni prefill e in ogni token.  Non era
coperto da nessun test (il batch era verificato solo per IQ2_S) e `391` passava
comunque perche' il MoE somma 10 esperti su 512 e l'errore si diluisce.
Corretto; ora **bit-identico** al kernel a token singolo.

**2. Copertura dei kernel di produzione CHIUSA.**  Aggiunti
`check_batch_any()` (5 tipi q8k) e `check_fp32_batch()` (down IQ4_NL/Q2_0, k=640 e
2560): **tutti e 7 i kernel batch ora danno `0.00e+00`** contro il percorso a token
singolo, che e' validato contro un riferimento indipendente in doppia precisione.
Da eseguire **sempre** dopo ogni modifica:

```sh
make test-qwen4-cpu-dot && ./tests/test_qwen4_cpu_dot | grep -E "vs singolo|fp32 batch"
```

**3. Mappa completa del modello ISTA** (layer MoE per tipo, rate L1-hot stabile):

| tipo | kernel | layer | L1-hot | streamed |
|---|---|---|---|---|
| IQ2_S | VNNI | 10 | 26,8-30,6 | 6,9-13,0 (oscilla) |
| **IQ3_S** | maddubs | **13** | **14,9-15,1** | 8,8-10,3 |
| IQ2_XXS | maddubs | 9 | 23,0-23,7 | 8,9-9,4 |
| IQ2_XS | maddubs | 10 | 21,9-26,9 | 8,8-9,9 |
| IQ3_XXS | maddubs | 6 | 21,4 | 10,0 |
| IQ4_NL / Q2_0 | fp32 (down) | 18 / 30 | — | — |

Carico **identico** a UD (T=1739, ne=512, ns=10, k_in=2560, k_ff=640, pairs=17390,
56,98 GMAC/chunk): il divario di velocita' e' tutto nel **tipo**, non nel carico
(UD usa IQ2_S con VNNI; ISTA ha formati con piu' lookup e catene piu' lunghe).

**4. Prossimo lavoro di velocita': riscrivere IQ3_S** (13 layer, unico fuori scala
a caldo: 14,9 contro 21-30).  Diagnosi: 512 MAC per passo in ~146 cicli = **0,8
istruzioni/ciclo** su una macchina che ne fa 3-4, quindi e' limitato dalla **catena
di dipendenze per passo** (indici -> 16 load scalari -> `set_epi32` -> `maddubs` ->
`madd` -> `add`), non dal throughput.  Il rimedio e' sovrapporre due passi (unroll
x2), ma con 8 token le coppie `sumi1`/`sumi2` occupano 16 ymm: serve prima liberare
registri, o con accumulo **int32 singolo + dpwssd**, o con **4 token per chiamata**.
Verifica: `DS4_BENCH_DOT=1`, battere 14,9-15,1 GMAC/s hot, poi `check_batch_any` e
`391` su ISTA prima di integrare.



---

## STATO E CONCLUSIONI (indice esecutivo - aggiornato a fine sessione)

**Numeri attuali** (macchina in buono stato; vedi regole di misura sotto):

| | prefill | note |
|---|---|---|
| UD-IQ3_XXS | **55-56,6 t/s** | da **5,5 t/s** a inizio campagna |
| ISTA GSQ-RCO IQ3_XXS | **45-49 t/s** | mid 540-550 ms su ~750 ms/chunk (72%) |

`391` corretto su **entrambi** i modelli, `ds4`/`ds4-server` 0 errori,
`test_qwen4_cpu_dot` verde, albero pulito (13 commit di sessione).

**Comando canonico per misurare** (UD; per ISTA cambia solo `-m`):

```sh
DS4_QWEN4_CPU_MOE=1 DS4_THREADS=16 DS4_QWEN4_CUDA_STREAM_EXPERTS=1 \
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=6 DS4_QWEN4_MOE_PROFILE=1 ./ds4 -m <gguf> \
  -c 16384 -ctk q8_0 -ctv q8_0 --cuda --prefill-chunk 1024 \
  --prompt-file /tmp/prefill_prompt.txt -n 1 --temp 0
```

**Tenuto** (con il guadagno misurato):

| modifica | guadagno |
|---|---|
| mid chunk-esterno (token-chunk fuori, righe dentro) | **+5,2%** end-to-end, bit-identico |
| prefetch NTA nel kernel **IQ2_S** (solo quello) | mid -2,5%, down -4% (t/s nel rumore) |
| silu8 vettoriale | ~2%; **incluse la correzione del bug** di store a 64 byte |

**Rimosso** (esperimenti negativi, ognuno con la misura che lo chiude):

| esperimento | misura |
|---|---|
| kernel a 2 righe nel worker | -12,6% sul mid |
| store non temporali nel mid | -7% |
| read-ahead `MADV_WILLNEED` dei pesi | -4% e +35% di disco |
| NTA sui kernel di ISTA | -3,7% (mid +5,3%) |
| staging L1 + flush contiguo | +1,2% = rumore |
| chunk da 16 token | +6% di tetto nel bench, rompe la residenza L1 |
| VNNI su IQ2_XXS/IQ3_S, B=16, accumulatore vettoriale | tutti negativi |

**Dove siamo, quantificato**: mid e down stanno all'**86-92%** e **76%** dei tetti
riprodotti nella loro stessa configurazione (bench `time_model_like_mt`,
`time_down_mt`: 152-166 GMAC/s a 16 thread).  Il margine residuo e' ~10% ed e'
**overhead del modello** (silu ~2,5% misurato, walk delle liste, indirezione
`tok_idx`), non kernel: il CPU-MoE e' al limite di questo design su questa
macchina.  Il prefill GPU e' 5,4 t/s (CPU-MoE 10x meglio): strada chiusa.

**REGOLE DI MISURA (imparate a spese di questa sessione)**:

1. **Solo A/B nello stesso run.**  A parita' di codice il prefill e' sceso da
   56,6 a 40,2 t/s nella stessa sessione (VM attiva da 5 giorni, ore di run,
   probabile throttling): i confronti fra run distanti non valgono nulla.
2. Prima di ogni campagna: `uptime`, `ps` (c'e' una VM `qemu-system-x86` costante
   all'88% di un core), `fincore` sul modello.  La VM **non** e' la variabile (e'
   costante); l'IO di pagina **non** e' sul cammino critico (3 run: 33 GB -> 55,6
   t/s vs 42 GB -> 56,6 t/s).
3. Il **bench misura un tetto, non predice l'end-to-end**: cinque modifiche
   "gemelle" su kernel della stessa famiglia si sono comportate in modo opposto.
   Ogni variante va misurata in-modello.
4. Metodo per l'IO di un run:

```sh
./ds4 ... > log 2>&1 & P=$!; LAST=0
while kill -0 $P 2>/dev/null; do V=$(awk '/read_bytes/{print $2}' /proc/$P/io); [ -n "$V" ] && LAST=$V; sleep 2; done
echo "$((LAST/1000000000)) GB letti"
```

**Hook in codice utili** (switch d'ambiente, tutti verificati):

| variabile | effetto |
|---|---|
| `DS4_QWEN4_MOE_COUNT=1` | stampa il carico reale per chunk (MAC, byte attivazioni, byte pesi) |
| `DS4_QWEN4_MID_DUP=1` | esegue i kernel del mid due volte: misura il peso sul cammino critico |
| `DS4_QWEN4_MID_NOKERNEL=1` | salta i kernel del mid (output spazzatura, solo per profilo) |
| `DS4_QWEN4_MID_ROWOUTER=1` | ripristina l'ordine riga-esterno (A/B del chunk-esterno) |
| `DS4_QWEN4_MID_NTA=0` | spegne il prefetch NTA (solo kernel IQ2_S) |
| `DS4_QWEN4_MID_SILU=0|1|2` | silu scalare / vettoriale / prodotto secco (riferimento) |
| `DS4_QWEN4_MID_STAGE=1` | staging L1 + flush contiguo (misurato = rumore, default spento) |
| `DS4_QWEN4_CPU_MOE_PREFILL=0` | prefill sulla GPU (misurato 5,4 t/s, solo per A/B) |
| `DS4_THINK=...`, `DS4_THINK_DEFAULT=...` | controllo thinking (server) |
| `--webui-dir DIR` | serve la webui llama.cpp dal server ds4 |

Bench isolati: `DS4_BENCH_DOT=1 ./tests/test_qwen4_cpu_dot` (kernel singoli, batch,
`time_model_like`, `time_model_like_mt`, `time_down_mt`, `check_batch2`).

---


Stato: branch `feat/qwen38-unified` (27B validato a 51 tok/s). Questo doc dice
cosa manca per far girare Flash-Next (MoE 176B) su ds4 su QUESTA macchina.

## Cosa esiste gia (main, PR #991)
- Grafo dedicato Metal/CUDA `qwen4exp`: gated delta-net, GDN, GQA con indexer,
  block-sparse attention, hyper-connections, n-gram embeddings, MoE, MTP.
- Tabella n-gram (95 GB) su SSD a righe, senza preload. YaRN x2/x4 oltre 262K.
- Server alias `qwen3.8-flash-next{,-chat,-reasoner}`, batching Metal nativo.
- Numeri Spark 128 GB: prefill 516-755 tok/s, decode ~22 tok/s.

## Decisione (17 set): supportare i NOSTRI file in ds4 (opzione B)

Niente download recipe da 137 GB: si adatta ds4 al packing Unsloth gia
su disco e validato. Il recipe resta piano di riserva/riferimento.

### Percorsi modelli (non spostare: router, script e preset li usano)
- `~/models/qwen38-flashnext-gguf/UD-IQ3_XXS/` (77 GB, default) — 3 shard
- `~/models/qwen38-flashnext-gguf/UD-IQ4_XS/` (88 GB, qualita max) — 3 shard
- `~/models/qwen38-flashnext-gguf/IQ3_XXS/` (71 GB, ISTA GSQ-RCO) — 2 shard
- `~/models/qwen38-flashnext-gguf/MTP/` (teste draft 1.8-2.6 GB)
- `~/models/qwen38-flashnext-gguf/imatrix_unsloth.gguf_file` (554 MB)
- `~/models/Qwen3.8-27B/` (27B dense + mmproj + MTP in-file)
- Symlink router: `~/models/Flash-Next-{ISTA,UD}-*` -> sottocartelle sopra
- Config: `~/.config/qwen38-router.ini`, script `~/bin/qwen38-flashnext`

### Lavoro stimato per B (raffinato 17 set dopo survey header GGUF)
1. Nomi: NESSUN mapping (Unsloth usa gia lo schema `qwen4exp`).
   Resta: loader multi-shard (shard 1 = solo KV) + shape no-MTP
   (block_count 48, nextn=0) + validazione allargata ai nuovi tipi.
2. Kernel MoE: down a 640 fisici, NESSUN pad al load (il pad rompe
   mmap-no-copy e mangia il vantaggio taglia). I kernel devono gestire
   width 640 nativa; il check layout e gia generico (pad solo per Q2_K).
3. Gather PLE con dequant IQ4_NL on-read (strada nuova; riga da 90 B).
4. MTP: nessun blocco nextn in-file -> niente speculazione fase 1;
   `--mtp` rifiuta pulito. Teste `MTP/` fuori scope fino a trunk validato.
5. Validazione numerica completa (probe 391/50x/300 + drift).

## Gap 1 (risolto dalla decisione: si fa B, A resta riserva)
I file Unsloth usano gia nomi/tensori dello schema `qwen4exp` di
llama.cpp (stessi `blk.*` di ds4), quindi il lavoro e su tipi/shape,
non sui nomi. La recipe ds4 (`download_model.sh qwen38-q2`, 137 GB:
trunk+MTP+n-gram BF16, down Q2_K paddati 640->768) resta solo piano
B di riserva. Survey 17 set dei file su disco (header GGUF letti con
gguf-py di llama.cpp master):

### Matrice quant (verificata 17 set, layer 0 + tensori globali)

| classe tensore | recipe q2 | UD-IQ3_XXS (default) | UD-IQ4_XS | ISTA GSQ-RCO |
| --- | --- | --- | --- | --- |
| token_embd / output | Q8_0 | Q6_K | Q8_0 / Q6_K | IQ3_S / Q5_K |
| dense (attn, ssm_out, shexp) | Q8_0 | Q6_K | Q8_0 | IQ4_XS/IQ3_S/Q4_K/Q5_K/Q6_K * |
| gate/up routed | IQ2_XXS | IQ2_S | IQ3_S / IQ4_XS * | IQ2_XXS/IQ2_XS/IQ2_S/IQ3_S/IQ3_XXS * |
| down routed | Q2_K pad 768 | IQ4_NL 640 | IQ4_NL / Q8_0 * | IQ4_NL / Q2_0 * |
| indexer q/k | bf16 | BF16 | BF16 | BF16 |
| PLE `per_layer_token_embd` | BF16 | IQ4_NL | IQ4_NL | IQ4_NL |
| norm/bias/router | F32 | F32 | F32 | F32 (alcuni BF16) |

`*` = Unsloth Dynamic: il tipo cambia da layer a layer (importanza).
Comune a tutti: 48 layer (no blocco MTP in-file), split 2-3 shard
(shard 1 = solo KV, 0 tensori), down NON paddati (640 fisici),
`ffn_gate_inp_shexp` 1-D (gia gestito), PLE `[160, 320001536]`.

### Nuovi tipi da supportare (ordine: default prima)
1. Fase 1 (UD-IQ3_XXS): Q6_K dense+embed+head, IQ2_S expert,
   IQ4_NL expert + PLE gather on-read. IQ4_NL non esiste in ds4
   (nemmeno `DS4_TENSOR_*=20`, typeinfo placeholder): blocco nuovo.
2. Fase 2 (UD-IQ4_XS): + IQ3_S/IQ4_XS expert (dense gia Q8_0).
3. Fase 3 (ISTA): + Q2_0 (nuovo), Q4_K/Q5_K dense, IQ2_XS/IQ3_XXS
   expert, IQ3_S embed, Q5_K head.

Lato CUDA gran parte passa per `cuda_matmul_mmq_dense_quant` (MMQ
ggml-derived: Q6_K/IQ2_S gia instradati, IQ4_NL da verificare in
`cuda/mmq`); MoE expert-GEMM, embed-gather e PLE-gather vanno
controllati uno a uno. Lato Metal (non testabile qui, niente Mac):
rifiuto esplicito dei tipi Unsloth, path recipe invariato.
Lato CPU-ref: riuso dequant Q6_K/IQ2_S esistenti, nuova dequant
IQ4_NL (banale: nibble su grid fissa + scala f16).

## Gap 2: budget VRAM (16 GB) — numeri reali 17 set
Pesi UD-IQ3_XXS ex-tabella: **49.5 GB** (denso ~4.2 + esperti ~45.3;
la tabella PLE da 26.8 GB resta su SSD). Il preload CUDA vuole cacheare
OGNI tensore su device: oltre ~15 span da ~1 GB la 16 GB finisce (provato:
OOM allo span 16 dopo 15 GB cacheati). La recipe q2 (41.7 GB resident)
non sarebbe stata diversa: il preload-tutto non scala oltre la VRAM.
Strada: `DS4_CUDA_DIRECT_MODEL=1` (salta il preload, pesi on-demand con
register per-range + staging shard-aware) invece di SSD-streaming (non
implementato per qwen4). KV 32K q8_0 ~1.6 GB + stati ricorrenti fissi
(GDN ~113 MB) + workspace: budget stimato 3-5 GB, ci sta.
Ricetta di partenza: ctx 32768, KV q8_0, niente draft, DIRECT_MODEL.
Riferimento llama.cpp stessa macchina: IQ3_XXS 18-19 tok/s @32K.
Obiettivo ds4: sopra (quanto sopra lo dice la misura, non l'assunto).

## Gap 3: verifica (regola del repo: correttezza prima della velocita)
- First-token test contro riferimento CPU su prompt fissi (391, 50x, 300).
- Drift check su decode lungo (i nostri probe A/B esistono gia).
- Memcheck dei nuovi path (precedente: batching Metal documentato cosi).

## Non serve (aggiornato per B)
- Nuovo backend CUDA: il grafo qwen4exp esiste gia, niente flag speciali.
- MTP esterno/in-file: i file Unsloth non hanno il blocco nextn -> fase 1
  senza speculazione; i 3 GGUF in `MTP/` restano fuori scope finche il
  trunk non e validato. `--mtp` con questi file deve rifiutare pulito.
- Speculazione ngram: misurata inutile in offload su llama; rivalutare qui.

## Diario slice
- 17 set B1 (loader multi-shard): shard 0..N rimappati in un'unica
  reservation VA contigua (MAP_FIXED, page-pad, zero copy) cosi tutto il
  codice map+offset resta invariato. Merge directory con dup-check +
  verifica split.no/count/tensors.count; tabella PLE anche mid-file
  (hole-punch a pagine intere, lettura via pread con file_offset di shard).
  Prova su UD-IQ3_XXS reale (77 GB, 1224 tensori): bind completo, stop
  atteso al gate BF16 (lavoro B5). Test: `test-qwen4-split` (nuovo),
  `test-qwen4-ngrams` (tolleranza tabelle disallineate), `test-deepseek41-gguf` invariato.
- 17 set B2 (shape no-MTP): NESSUN codice nuovo necessario. Il trim
  `nextn_predict_layers` (main, 12 set) riduce gia il profilo a 48/0 quando
  la chiave manca (caso Unsloth) e tiene 49/1 per la recipe; `--mtp` con
  NEXTN=0 rifiuta pulito in `ds4_engine_open_internal` (config prima del
  rifiuto). Bloccato con `test-qwen4-config` (nuovo): fixture 48 senza
  chiave -> 48/0, fixture 49 con nextn=1 -> 49/1, predicati layer e
  steering-count in entrambi i profili.
- 17 set B3+B5 (tipi CPU-ref + PLE IQ4_NL): `block_iq4_nl` + dequant
  speculare a llama.cpp (low nibble first, grid verificata identica a
  upstream) + vecdot; casi Q6_K/IQ2_S/IQ4_NL in `qwen4_ref_row` (copre
  dense, MoE, embedding, head); allow-list dense+expert; gate PLE e
  `qwen4_ngram_row` accettano IQ4_NL (riga 90 B, riuso dequant). Prova su
  UD-IQ3_XXS reale: `model_open` OK (3 shard, 1224 tensori, 76.33 GiB),
  down blk.47 = 471859200 B esatti (=838860800/32*18), tabella aperta via
  fd di shard. Test: `test-qwen4-dequant` (nuovo: oracle indipendente +
  hand-cases esatti + cross-check dot), tabella IQ4_NL in `test-qwen4-split`.
- 17 set B4 (kernel CUDA, solo build): reader `value<>` per 14/22/20 +
  `scalar` 14 + `row_bytes` + dispatch matvec/moe/unpack in
  `ds4_qwen4_cuda.cuh`; tabelle device `cuda_iq2s_grid` (copia verbatim
  host, verificata 1024/1024) + `cuda_iq4nl_values` in
  `ds4_iq2_tables_cuda.inc`; gate pesi esteso SOLO su build CUDA
  (`DS4_HAS_QWEN4_GPU`), Metal/ROCm/CPU invariati. PLE ed embedding
  restano host-side (nessun kernel). Prefill tiled escluso per i nuovi
  tipi (path per-token esatto). Compila (`ds4_cuda.o`, `ds4.o` sm_120);
  numerica device -> B7 (serve run GPU).
- 17 set B4-fix (primo boot): il pack e dinamico anche in UD-IQ3_XXS
  (layer 2 con gate/up IQ3_S, denso misto Q6_K/Q8_0). Aggiunto tipo 21
  ovunque (ref CPU, gate, `value<21>` device con grid `cuda_iq3s_grid`
  copiata verbatim, dispatch, test wiring+dot). Boot morto pulito al
  gate con messaggio esatto; server spento, VRAM a baseline, niente Xid.
- 17 set B4-fix2 (secondo boot): stop a `CUDA model range read failed
  ... EIO` + OOM apparente. Causa: lo staging CUDA fa `pread` sul solo
  fd dello shard 0 con offset globali (oltre 10 MB = oltre EOF). Fix in
  `ds4_cuda.cu`: tabella shard (`ds4_gpu_set_model_shards`, pubblicata da
  `ds4_gpu_publish_model_files` in tutti i path di engine-open),
  `pread` segmentato per shard, rewind dell'arena bump-allocator sui
  fallimenti (il leak mascherava l'OOM). Label n-gram ora type-aware.
- 17 set B4-fix3 (terzo boot): preload-tutto non scala oltre la VRAM
  (pesi ex-tabella 49.5 GB: denso ~4.2 + esperti ~45.3; OOM allo span 16).
  Strada: `DS4_CUDA_DIRECT_MODEL=1` (pesi on-demand, niente preload).
  Gap 2 riscritto con i numeri reali.
- 17 set B4-fix4 (prompt no-think): `matmul failed` per denso Q6_K con
  8<T<32 tok: `dense_mm` copre T<=8 (matvec), T>=32 (unpack+cublas) e
  `matrix_dispatch` per gli altri, ma a 14 mancava il caso. Aggiunto
  `value4<14>` (4 letture scalari, valido a ogni i) + `QWEN_TC(14)`.
- 17 set B7-391-OK: `17*23=?` no-think, 200 tok: **391 esatto** con
  passaggi intermedi giusti (340+51, due metodi). Prima prova che
  MoE+PLE+head device coi nuovi quant calcolano bene. Decode ~1 tok/s
  (paging-bound). Server caldo su 8892.
- 17 set B7-ab-math: harness documentato 3/3 OK (266, 49, 784) come
  llama UD, ~36 min totali (857+767+477 tok thinking). Zero errori device.
- 17 set B7-bench (chiuso): prefill corto ~2.1 tok/s, decode stabile
  ~1.0-1.15 tok/s (109 tok risposta + 500 tok thinking, flat). Riferimento
  llama: 18-19 tok/s -> divario ~18x, tutto paging on-demand (GPU idle).
  Probe 50x/300 liberi superati da ab-math; sanitizer non disponibile.
  Server spento, VRAM libera.
- 17 set B8.0 (survey MTP): 3 file, arch qwen4exp, singolo blocco
  `blk.48` (full-attention + indexer + MoE + hc, niente GDN/PLE) +
  testa nextn (eh_proj [5120,2560], enorm, hnorm, hc_head down/up/norm).
  `Q4_K_M` standalone (token Q4_K + head Q6_K); `shared-*` senza
  embed/head (usano il trunk). `shared-Q8_0` non chiede tipi nuovi
  (tutto Q8_0/BF16/F32) -> validazione prima con quello; `Q4_K_M`
  chiede Q5_0 (hc_up, tipo nuovo ovunque). Draft ~= 1/48 target + testa.
- 17 set BUG-P0 request-2 (APERTO, workaround: fresh boot per misura):
  dopo un run lungo (>=~120 tok) il request successivo degenera (`!`
  da subito o dopo pochi token), deterministico, prompt-indipendente.
  ESCLUSO con prove: pesi (FT_LIST 64/64 + dump A/B identici), prompt
  (id byte-identici), sampling (temp-0), sampling-path (plain eval),
  fd-cache (NO_FD_CACHE riproduce), reset mancato (trace: RESET gira),
  thinking-split (falsificato), tabelle/grid (verificate). Reset esteso
  a k/v/ik/block_key non risolve. Indizi: onset anticipato con arming
  lungo; temp-1.5 resta `!` (logit `!`-dominante, non flat); UD3 mai (8 run).
  Sospetto residuo: stato sessione/server fuori dal graph o interazione
  live-checkpoint. Traccia TEMP-DEBUG in sync() da rimuovere al fix.
  18 set (P0, caccia): `!` = token 0 = vincitore spareggio argmax -> logit
  piatti/NaN. Trap TEMP [tmp-flat] armata (tiene, costo ~zero): collasso
  mid-run a pos 147 dopo `...oplasts** of` + prefill req2 piatto a pos 17
  (stesso prompt su fresh boot e sano). File righe finite (CPU-ref),
  reset completo a campioni, VRAM piatta, zero errori/OOM/fallback.
  Stesso prompt+temp0 diverge tra boot (race/layout); UD3 mai (10 run).
  Esclusi: prefill lungo (1159 righe sano), cumulativo multi-request
  (396 tok corti sani), prompt/pesi/sampling/fd-cache. Sospetti: OOB
  kernel che inchioda stato device fuori reset, reduction non deter-
  ministica + cliff numerico, aliasing pool. Workaround resta fresh boot.
- 18 set P0-caccia (3 punti collasso): pos 147 dopo `oplasts** of`,
  pos 128 dopo `Question: How many`, pos 131 dopo `O _ 2 $$` — cluster
  128-147 + token banali => drift-to-cliff, non content-trigger.
  Niente atomic float (solo istogrammi int) => compute deterministico
  => divergenza da input diversi: uninitialized-read (garbage boot-
  dependent) il sospetto ora; sanitizer assente. Trap [tmp-flat]
  resta armata (allarme a costo zero con trigger).
  ~1.0-1.15 tok/s (109 tok risposta + 500 tok thinking, flat). Riferimento
  llama: 18-19 tok/s -> divario ~18x, tutto paging on-demand (GPU idle).
  Ottimizzazioni (resident-dense, value4/tiled, prefetch) dopo la qualita.
  Probe 50x/300 liberi superati da ab-math; sanitizer non disponibile.
  Server spento, VRAM libera.
- 17 set B6-impl (UD-IQ4_XS + ISTA, solo build): alla domanda perche non
  prendere da llama.cpp, verifica: quasi tutto esisteva gia (dequant CPU
  dal 27B, MMQ ggml-derived, grid verificate) — restava da cablare.
  Nuova matematica solo Q2_0 (blocco+dequant speculari a ggml,
  00=-1..11=+2, LSB-first). Aggiunti: ref CPU per IQ2_XS/IQ3_XXS/IQ4_XS/
  Q5_K/Q2_0, validazione router BF16, reader device 13/17/18/23/42 +
  value4/scalar/dispatch, gate CUDA. Parse OK su entrambi i file reali
  (87.25 e 70.63 GiB). Test dequant esteso. Validazione GPU: da fare.
- 18 set B6-ISTA (boot+391): dense gate +Q5_K/IQ3_S/Q4_K/IQ4_NL/IQ4_XS/Q2_0,
  dense_ok +Q4_K/IQ4_NL, QWEN_MV +20/+42, hc up/inject BF16 solo CUDA
  (check + hc_gate_mix/hc_norm_prefill<30>; Metal resta f16/q8).
  Trovato con trap NaN per-stadio: moe slot legge gate/up con un solo
  tipo/stride -> con shexp misti (gate IQ4_XS, up IQ3_S) legge male e
  NaN; fix: shared expert via dense GEMV quando gate/up differiscono
  (path Metal-prefill gia testato; pack uniformi invariati). ISTA boota,
  391-probe 60 tok sani (~1 tok/s), zero NaN. UD3/UD4 invariati (regr).
  Test CUDA TC23 (fork-isolato, 4 shape). Resta: ab-math ISTA/UD4.
- 18 set PERF breakthrough (cache-mode): DS4_CUDA_DIRECT_MODEL bypassa
  la range-cache (tutto riletto da host ogni riga: ~6 GB/riga PCIe +
  stalli page-cache/SSD -> ~1 tok/s). Senza direct_model la cache riempie
  (dense residente + expert fill-on-demand): UD3 60 tok 25s (4.1 tok/s),
  200 tok 41s (5.4 tok/s), zero flat. Con LIMIT_GB=10: 200 tok 48s
  (~4.2 tok/s), VRAM 12.9/16 (headroom), 1x "cache full" poi fallback
  graceful. Config da ora: STREAM_EXPERTS=1, no direct, LIMIT 10.
  Prossime leve: LRU/eviction (ora fill-forever first-touched),
  hot-expert pinning (top imatrix caricati per primi), overlap,
  MTP da rivalutare a base 4 tok/s, P0 per affidabilita long-run.
- 18 set pinning AB negato: static top-2723 (30.8% massa) pinna 9.2 GB
  ma single-topic 200 tok 65s vs 41s nopin, multi-topic 124s vs 104s
  (crowding del working set adattivo). Revert a fill adattivo; resta
  LRU/eviction dinamica come direzione (non statica).
- 18 set row-profile decode (MOE_PROFILE esteso al path row T=1, tooling
  env-gated tenuto): per-layer router 0.14 + mid 8.6 + down 6.6 +
  reduce 0.02 ms; MoE ~190ms/tok caldo vs pavimento HBM ~1-2ms (~100x:
  1152 micro-GEMV latency-bound, ~1GB/s effettivi). Router/reduce
  trascurabili. Cache mai piena a 10GB nei test (LRU non serve ora).
  MTP a base 4 tok/s: 150 tok 47.5s vs 38s (-25%), output identico
  (verify esatta) ma draft head (145ms) > budget (~22ms a p=0.9,
  trunk 222ms). Parcheggiato finche draft non costa <20ms.
  Leva regina: MoE fuso/batched (persistent kernel, 3-5x realistici);
  poi overlap; MTP solo con draft economico.
- 18 set DISEGNO MoE (analisi, no GPU): moe_mv e gia 1 launch/stage
  (grid 160x11x1, 1 warp/riga, dot K=2560 on-the-fly). Ma UD3/UD4/ISTA
  usano tipi 18/23/21/42 = path SCALARE (value<>/dot elemento per
  elemento, ~25 op seriali); il path float4 vettoriale copre solo
  10/12/16/39 (il Q4_K del recipe non scaricato: ironia opzione B).
  Aritmetica: 1760 warp x 128K op = 225M op/mid/layer -> ALU-bound
  confermato (~1GB/s effettivi vs 896 HBM). Piano A (giorni):
  value4<18>/<23>/<21> vettoriali veri (fetch superblocco condivisi
  per 4-pack: d/aux una volta, 4 lane parallele) + gate nel path
  float4 di moe_mv. Stima 4x MoE -> riga 190->~80ms -> 10-13 tok/s.
  Verifica: A/B vs CPU ref (qwen4_ref_row) + FT_LIST + abshort +
  P0-watch. Piano B: fusione mid+swiglu+down; Piano C: persistent
  MoE (settimane). Prefill tiled `matrix` gia ammortizza (non urgente).
- 18 set value4<18> NEGATO e revertito: 4-pack vettoriale (fetch
  condivisi) +40% mid e +36% down su decode pulito pos>0 (4.55->6.26
  ms/layer, 576 campioni A/B stesso prompt). Il compilatore CSEa gia
  i fetch ridondanti via L1; il branch aggiunge registri -> occupancy
  giu. Lezione: i numeri 8.6/6.6 erano inquinati dal prefill (grep
  "T=1" matcha "T=18"!); baseline vera decode MoE 4.55ms/layer
  (router .03/mid 2.49/down 2.01). Collo vero: LUT dipendenti +
  occupancy, non fetch ridondanti. Prossimo: microbench sintetico
  per componenti separati, oppure accettare MoE e spingere overlap.
- 18 set SVOLTA ARCHITETTURALE: CPU-MoE (path stile llama.cpp auto-fit, opt-in
  DS4_QWEN4_CPU_MOE=1). Misure che l'hanno imposta, tutte riprodotte:
  PCIe = Gen3 x16 (nvidia-smi: Device Max 5, Host Max 3) -> ~12 GB/s
  pratici; SSD O_DIRECT 2.58 GB/s; page cache del modello 44.6% (shard 2
  appena 21%) con 62 GB RAM e 76 GB di modello; riferimento llama.cpp
  MISURATO ORA stesse condizioni (VM Windows accesa): 17.9 tok/s decode
  @30K (55.8 ms/token), 177 tok/s prefill, e il log conferma "tensor
  overrides to CPU are used with mmap enabled" = esperti su CPU. Fisica:
  0.96 GB/token di esperti attivi (10/512 x 48 layer, parser GGUF:
  gate/up IQ2_S/IQ3_S, down IQ4_NL) -> 18 tok/s = 17.2 GB/s: impossibile
  su PCIe Gen3, banale su DDR4 (misurata 40 GB/s con 8 thread).
  Implementazione: kernel AVX-512 per IQ2_S (tabelle f32 precalcolate di
  griglie+maschere segno), IQ3_S, IQ4_NL (validati vs dequant scalare,
  worst 8.3e-6, test tests/test_qwen4_cpu_dot.c), driver parallelo sul
  pool esistente, readback pinned (ds4_gpu_host_alloc), shared expert
  ancora su GPU come proiezioni dense, reduce GPU invariato.
  RISULTATI: 391 TRUE, abshort 6/6, 200 tok a 10.35-10.72 tok/s steady
  (da 4-5), CPU-MoE 65.5 ms/token (16 thread; 71.8 con 8), read 0.014,
  write 0.021. Profilo: encode 91.9 ms, gpu 2.46 ms -> la GPU e' idle al
  97%. Collo reale: kernel CPU issue-bound (2.91 GB/s/core L1-hot,
  2.70 da RAM) e 26 ms/token di dispatch/launch. Non e' la RAM (40 GB/s).
  Prossimi passi misurati: (A) kernel int8/maddubs per IQ2_S (strada
  ggml: ~2x per core, atteso ~40 ms/token), (B) split ibrido: gli hot
  expert imatrix in VRAM li calcola la GPU idle (~9-10 GB utili = ~20-30%
  delle selezioni), (C) tagliare i 26 ms di dispatch. Con (A)+(B) il
  budget torna sotto i 55.8 ms di llama.
- 18 set CPU-MoE ibrido TENTATO e RIMOSSO (esperimento negativo documentato):
  pin dei top expert imatrix in VRAM (2723 = 5 GiB, pin verificato) + split
  per slot (16% su GPU idle). Risultato: 75.4 ms/token contro 65.5 del
  CPU-only, e output SBAGLIATO (340+51=341) -> non spedito. Cause
  diagnosticate: (a) il write async/ordine con i kernel per-slot non e'
  affidabile come pensavo, (b) il costo di scansione della range-cache
  cresce con le migliaia di range pinnati (attribuito al bucket kernel),
  (c) l'arena sovradimensiona (8 GiB pinned -> OOM a budget 13). Lezione:
  ogni byte spostato su GPU va pagato in dispatch, e il guadagno va
  misurato, non dedotto. Codice rimosso, resta il CPU-only validato.
  NOTA: l'API di query residenza aggiunta e poi rimossa; se si riprende
  l'ibrido, partire dal costo di scansione della cache, non dal pin.
- 18 set VALIDAZIONE LONG-FORM CPU-MoE (ab-math completo, 3 problemi,
  3000-4000 token max, ctx 32K, boot fresco): p1 266 OK (857 tok, 110s),
  p2 49 OK (767 tok, 100s), p3 784 OK (477 tok, 74s su boot dedicato).
  Decode sostenuto 11.5-13.5 tok/s su generazioni da 3000 token (primi
  numeri long-form del path CPU; nessun degrado fino a 3K di contesto).
  INCIDENTE P0 COLPITO E CARATTERIZZATO (3/4 sessioni pulite): nella run
  sequenziale il 3o problema, subito dopo "thinking live checkpoint
  remembered ... live=857" + "sync RESET", ha prodotto logits ±inf al
  ULTIMO token del prefill (pos=92 di 93) -> 3000 token di spazzatura ->
  NO-ANSWER. La trappola [tmp-flat] ha registrato 3001 eventi
  (finite=0/248320, minmax=[inf,-inf]): niente corruzione silenziosa.
  La richiesta successiva (prompt nuovo) e' tornata corretta (144), e un
  boot fresco con lo stesso p3 ha dato 784 senza un solo trap: il difetto
  NON e' nel path CPU ne' nei kernel - e' nello stato di sessione del
  server (percorso live-checkpoint/reset). Ricetta di riproduzione ora
  concreta: (1) generazione lunga ~800 token, (2) prompt nuovo ~90 token,
  (3) reset -> primo forward con logits non finiti. Prossimo passo P0:
  ispezionare cosa il reset NON azzera (stato GDN/conv, storia PLE,
  tabella block-key, checkpoint live) invece di cercare nei kernel.
- 18/19 set P0 CACCIA (strumenti + evidenza, ricetta ora riproducibile):
  aggiunta scansione di stato ([tmp-state], one-shot sul primo trap, gated
  DS4_QWEN4_STATE_SCAN per il post-reset) + trap [tmp-flat] esteso a
  mixed/R/blk/part/mid/router/ple_hist/ple_emb/pos3/cache/scratch
  attention (score/tile_max/attn_part/sel_blocks/sel_tokens/moe_lists/n_sel).
  FATTI MISURATI: (1) post-reset lo stato e' PULITO (6/6 scansioni, nessun
  non-finito) -> il reset NON e' il colpevole; (2) il trap riproduce al
  2o ciclo di una sequenza A/B (generazione 500-600 tok + prompt nuovo
  ~30-93 tok con reset), 2 volte su 3 tentativi, 0-2 su altre sequenze;
  (3) quando esplode, la cascata parte a META' chunk (ik_cache nan da
  posizione 7, k_cache da 14) e poi avvelena tutto a valle (mixed/R/blk/
  router/part/mid nan); n_sel e' zero, quindi non e' la selezione top-k;
  (4) gli n-gram/PLE e le posizioni restano finiti.
  SOSPETTO PRINCIPALE, nel codice: `block_key` in ds4_qwen4_cuda.cuh fa il
  pooling su `ratio` righe SEMPRE (v[i] = a/ratio), senza clampare alle
  righe davvero scritte nel chunk: con l'ultimo blocco parziale legge
  righe oltre il chunk (dopo un reset sono zeri -> chiave sbagliata ma
  finita; con righe residue non azzerate -> valori anomali). Va verificato
  se la riga 7 (ultima del primo blocco con ratio 8) entra nan dall'ik
  scritto o dal pooling. Esperimento proposto: variante gated che azzera la
  coda del blocco parziale prima del pooling, A/B su 3 tentativi della
  ricetta. NON toccato il modello: ogni fix va validato (ab-math + abshort).
- 19 set P0: POSIZIONE ESATTA DELLA PRIMA NaN. Dimensioni: n_head_kv=2,
  n_head_dim=256 -> kv_dim 512; n_indexer_head_dim=128. Le tre tracce
  coincidono: k_cache prima NaN a float idx 1792 -> posizione 7
  (1792/(512/2)); ik_cache a 896 -> posizione 7 (896/128); R a idx 71680
  -> riga 7 (71680/(4*2560)). Chunk da 30 righe con ratio 8: la posizione
  7 e' l'ULTIMA riga del primo blocco pooled completo, cioe' il primo
  punto in cui il path attention sparsa (block key + selezione) puo'
  attivarsi su un blocco intero. Ingressi (ple_emb, ple_hist, pos3) e
  stato pre-forward risultano TUTTI finiti (scansioni post-reset 6/6
  pulite) -> la NaN nasce DENTRO il layer in quella posizione, non da
  stato stantio del reset ne' dai pesi (altre richieste sullo stesso boot
  sono corrette). Conseguenza: la coda "stantia" del blocco parziale NON
  spiega la NaN (dopo il reset quelle righe sono zere) - il sospetto si
  sposta sullo scratch dell'attention sparsa (score/tile_max/attn_part/
  sel_blocks/sel_tokens/n_sel), che il reset non azzera e che al primo
  blocco completo viene letto per la prima volta. Fix candidato (da
  testare, A/B sulla ricetta): azzerare quello scratch nel reset, cosi'
  una sessione riusata si comporta esattamente come un boot fresco.
  Ricetta: generazione 500-600 tok + prompt nuovo 30-93 tok con reset;
  riprodotto 2 volte su 3 tentativi (trap [tmp-flat] 562-662 eventi).
- 19 set P0: ipotesi "scratch sparsa stantia" INDEBOLITA dal codice: nel
  prefill (cT > 2) l'attention usa i blocchi densi e passa attn_part = NULL
  (62287/62313), quindi la selezione sparsa non e' nemmeno letta nel chunk
  che fallisce; inoltre al trap n_sel risultava gia' 0. Fix speculativo
  (azzerare lo scratch nel reset) NON applicato: senza evidenza non si
  cambia la matematica. Estesa invece la scansione [tmp-state] ai prossimi
  candidati: block_key per layer (chiavi pooled) + q/iqn/attn_o
  (ingressi/uscite attention). Binario ricompilato e verificato sano
  (output coerente, 0 trap). PROSSIMO PASSO: far scattare la ricetta e
  leggere la scansione - se block_key e' NaN con ik_cache finita il
  colpevole e' il pooling (block_key in ds4_qwen4_cuda.cuh, media su
  `ratio` righe senza clamp); se q/iqn sono NaN il difetto e' a monte,
  nell'attention densa alla prima posizione con blocco completo.
- 19 set P0: tentativo di discriminazione con scanner esteso (block_key,
  q, iqn, attn_o) NON riprodotto: 4 cicli puliti (8 reset) contro 2
  riproduzioni su 3 tentativi in precedenza -> tasso reale ~1 su 4-6
  cicli, non 1 su 3. Strumentazione pronta e a costo zero quando non
  scatta (gated/one-shot); prossima occorrenza dira' se la NaN nasce nel
  pooling dei block key o nell'attention densa a monte. Nota di metodo:
  con questo tasso servono 3-4 sessioni di ricetta per avere un dato,
  pianificarle di conseguenza.
- 19 set COMANDO CANONICO CPU-MoE (config validata: 391, abshort 6/6,
  ab-math 3/3, 10.5-13.5 tok/s):
  DS4_QWEN4_CPU_MOE=1 DS4_THREADS=16 DS4_QWEN4_CUDA_STREAM_EXPERTS=1 \
  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=6 ./ds4-server -m <UD-IQ3_XXS-shard0> \
  -c 16384/32768 -ctk q8_0 -ctv q8_0 --cuda --port 8892 --prefill-chunk 1024
  Diagnostica: DS4_QWEN4_MOE_PROFILE=1 (tempi per fase), DS4_QWEN4_TIMING=1
  (forward), DS4_QWEN4_STATE_SCAN=1 (stato post-reset), trap [tmp-flat]/
  [tmp-state] sempre attivi. Test: make test-qwen4-config test-qwen4-cpu-dot
  test-qwen4-dequant test-qwen4-split test-qwen4-ngrams test-qwen4-drafthead
  test-cuda-tc23 CUDA_ARCH=sm_120.
  Prossime leve, in ordine di valore misurato: (1) [FATTO 19 set: voce
  sotto] kernel int8/maddubs per IQ2_S (~2x/core, atteso ~40 ms/token),
  (2) taglio dei 26 ms/token di dispatch (la GPU lavora 2.5 ms),
  (3) P0 con la ricetta quando si ripresenta.
- 19 set CPU-MoE INT8 IQ2_S (maddubs + AVX-512 VNNI): implementati in ds4.c
  accanto a `qwen4_cpu_dot_iq2_s_simd` il reference scalare
  `qwen4_cpu_dot_iq2_s_q8k_generic` (specchio di ggml generic),
  `qwen4_cpu_dot_iq2_s_q8k_maddubs` (AVX2, forma
  `ggml_vec_dot_iq2_s_q8_K`) e `qwen4_cpu_dot_iq2_s_q8k_vnni`
  (AVX512-VNNI, 64 valori/step: sign mask da vpshufb+test, `vpdpwssd`
  applica le sub-scale int16); dispatcher `qwen4_cpu_dot_iq2_s_q8k` che
  sceglie VNNI > AVX2 > generic. Le grid restano int8 (max 43, q8<=127:
  niente saturazione int16) e l'attivazione viene quantizzata UNA volta
  per layer in Q8_K (`ds4_quantize_row_q8_K`, scratch `xq` pinnato) e
  condivisa da tutte le righe IQ2_S di gate/up; il down IQ4_NL resta
  fp32. A/B diagnostico: `DS4_QWEN4_CPU_MOE_FP32=1` forza il path fp32.
  Validazione `make test-qwen4-cpu-dot` (esteso): i tre kernel sono
  confrontati a tolleranza 1e-5 contro un reference indipendente che
  dequantizza i blocchi Q8_K e somma in double (fase intera esatta;
  worst rel 6.3e-7..2.7e-6, loss di quantizzazione sintetica 0.5-14% su
  attivazioni uniformi random = caso peggiore). Microbench: `DS4_BENCH_DOT=1`
  IQ2_S fp32 2.60 -> q8k VNNI 4.79 GB/s single-thread (1.84x);
  `DS4_BENCH_RAM=1` 16 thread 256 MiB random 11.15 -> 15.61 GB/s (1.40x),
  8 thread 6.37 -> 9.59 GB/s (1.51x). Modello UD-IQ3_XXS, comando canonico
  + `-p "17*23=?" --nothink -n 80 --temp 0`: 391 esatto con q8k (340+51);
  A/B warm (`DS4_QWEN4_MOE_PROFILE=1`) mid 1.53 -> 0.97 ms/layer (1.58x),
  total MoE 2.33 -> 1.73 ms/layer (1.35x), down invariato; generation
  7.05 -> 9.10 tok/s nella coppia (macchina rumorosa: page cache/
  frequenza; il segnale stabile e' il mid, coerente col bench). Test
  restanti del set canonico verdi (config, dequant, split, ngrams,
  drafthead, tc23). Nota: il path cambia la numerica dell'attivazione
  (int8), stessa strada di llama.cpp per IQ2_S; 391 e i passaggi
  intermedi restano corretti. Leva residua: down IQ4_NL (~44% dei byte
  MoE ancora fp32, Q8_0+maddubs), poi dispatch 26 ms.
- 19 set ISTA CPU-MoE COMPLETO (kernel per i tipi Unsloth Dynamic):
  implementati i reference scalari + kernel AVX2 maddubs IQ2_XXS/IQ2_XS/
  IQ3_XXS/IQ3_S x Q8_K e un kernel fp32 AVX-512 Q2_0 (down ISTA),
  dispatch per tipo in `qwen4_cpu_row_dot_q8k`/`qwen4_cpu_row_dot`,
  allow-list estesa (16/17/18/42), stessa attivazione Q8_K per gate/up
  (verificato dal GGUF: gate==up su 48/48 layer ISTA).  IQ2_XS: la prima
  versione (estrazione scalare dei 7 bit di segno) faceva 1.64 GB/s;
  riscritta sulla forma ggml (128 valori/step, sign bit via
  shift+xor+shuffle) -> 4.24 GB/s.  `test-qwen4-cpu-dot` esteso a tutti
  i tipi (reference indipendente in double): worst rel IQ2_XXS 4.6e-7,
  IQ2_XS 9.7e-7, IQ3_XXS 3.2e-7, IQ3_S 3.0e-5 su casi di cancellazione
  (tolleranza 1e-4 come il path fp32), Q2_0 0.  Single-thread GB/s:
  IQ2_S 5.1 (vnni) / IQ2_XXS 3.1 / IQ2_XS 4.2 / IQ3_XXS 4.1 / IQ3_S 3.2 /
  Q2_0 3.8 / IQ4_NL 5.9.  RISULTATI modello (16 thread, comando canonico,
  200 tok, cache calda, -n 200): ISTA **4.11 -> 15.09 t/s** (MoE 42
  ms/token: mid 0.56 + down 0.28 ms/layer), 391 corretto; UD **13.24
  t/s** (MoE 48.5 ms/token), 391 corretto.  Confronto: llama ISTA
  14.37 (prima richiesta) - 17.07 (warm) -> ds4 a parita' sulla prima,
  ~12% sotto la warm; llama UD 14.89-21.85 -> UD ancora sotto.  GPU
  2.2 ms/token (idle); encode host 62.8 ms di cui 43 di MoE -> **~20-27
  ms/token di dispatch/launch** (ogni ds4_gpu_* lancia subito,
  `ds4_gpu_begin_commands` e' un no-op, `end_commands` e'
  cudaDeviceSynchronize: 48 sync/token).  Leva residua n.1: CUDA graph /
  fusione dei kernel per il tratto host; n.2 hot-set della page cache
  (838 MB/token ISTA, 950 UD); n.3 VNNI per gli altri tipi.
  `DS4_CUDA_END_STREAM_SYNC=1` provato: nessun guadagno chiaro
  (10.99 vs 13.24 su run singole rumorose).
- 19 set DISPATCH/CUDA-GRAPH REFUTATO dalle misure (spike chiuso
  negativo, da non riprovare): microbench su questa macchina - launch di un
  kernel vuoto 1.55 us (host), device sync 64 us, stream sync 0.2 us,
  memcpy 4K D2H 6.4 us.  Nel decode T=1 il profilo host per stage
  (`DS4_QWEN4_TIMING=3`, senza sync aggiuntivi) da': hc_attn 0.5 + gdn 1.0
  + attn 0.3 + hc_ffn 0.5 + ple 0.1 = **~2.4 ms/token di submission host**,
  non 20.  Con `DS4_QWEN4_MOE_PROFILE=1 DS4_QWEN4_MOE_PROFILE_NO_SYNC=1`
  il flush del CPU-MoE (attesa GPU per layer) e' **0.383 ms/layer = 18.4
  ms/token**, identico a 4K e 16K, con 1 o 16 thread e con stream-sync al
  posto di device-sync -> e' lavoro GPU reale seriale (GEMV decode
  latency-bound: ~30 kernel/layer, ~12 us l'uno), non pacing/dispatch.
  Niente CUDA graph quindi: il backend non batcha nulla
  (`ds4_gpu_begin_commands` e' un no-op) e catturare i segmenti attorno al
  CPU-MoE recupererebbe al massimo i 2.4 ms di submission, con rischio
  alto (parametri pos/pointer da aggiornare per replay).  Budget warm ISTA
  misurato: MoE 42 + GPU seriale 18.4 + host 2.4 ~ 63 ms/token = 15.9 t/s
  (15.09 misurati col sampling).  llama warm 17.07 (58.6 ms): il residuo e'
  GPU per-layer, non dispatch.  Leve ricalibrate: (1) fusione dei GEMV
  decode (o capture dei soli segmenti GPU) per ~16 ms; (2) VNNI dei kernel
  IQ2_XXS/IQ3_S/Q2_0 (~1-2 ms); (3) MTP rivalutato coi numeri - verify T=2
  raddoppia il CPU-MoE (~84 ms) + draft ~20 ms per ~1.6 token accettati =
  ~82 ms/token -> peggio, resta parcheggiato.  Diagnostica: TIMING=3 =
  stage host T=1; MOE_PROFILE_NO_SYNC=1 = flush GPU reale.
- 19 set shared expert in overlap col CPU-MoE: le proiezioni dense dello
  shared expert (dipendono solo da `mixed`) vengono accodate subito dopo le
  readback, prima dei dot paralleli, cosi' la GPU lavora mentre la CPU
  calcola gli esperti; il caller salta il suo blocco shared per il path CPU.
  `flush` reale 0.383 -> **0.349 ms/layer (~1.6 ms/token)** misurato con
  `MOE_PROFILE=1 MOE_PROFILE_NO_SYNC=1`; end-to-end dentro il rumore
  (14.75-14.84 vs 14.96-15.09 t/s su ISTA warm, macchina variabile), 391
  corretto.  Tenuto: riduce l'attesa GPU senza cambiare la semantica
  (stesso ordine di stream).
- 19 set MMVQ sui dense qwen4 (kernel GPU, chiuso il collo n.1): il
  profiler GPU a eventi (`DS4_QWEN4_TIMING=4`, nuovo: mark su stream 0,
  report per-stage ms/token) ha mostrato che ~10 dei ~18 ms/token seriali
  stavano in DUE GEMV per layer GDN (qkv+z 6.4 + ssm_out 3.0 ms) e nelle
  proiezioni attention (3.7 ms), tutti serviti dal path scalare `value<>`
  (i tipi ISTA 13/21/23 sono deleghe scalari per scelta documentata).
  `ds4_gpu_qwen4_dense_mm_tensor` ora instrada T<=8 con K%%256==0 e tipi
  {12,14,16,17,18,19,21,22,23,29} su `ds4_mmq_quant_dense_vec` (MMVQ
  upstream, gia' vendored in cuda/mmq e usato dal path qwen38); IQ4_NL/
  Q2_0 (K=640) e Q8_0/F16/BF16/F32 restano sui kernel locali.  A/B stesso
  boot (`DS4_QWEN4_NO_MMVQ=1`), flush reale 0.345 -> **0.214 ms/layer**
  (~6.3 ms/token), stage GPU gdn pair 6.4 -> 2.1, lin_out 3.0 -> 0.92,
  attn 3.7 -> 1.43 ms; generazione 13.9 -> **15.9 t/s** nella coppia.
  Warm 200 tok ISTA: 15.01/14.89/12.85 (rumore page cache), 391 corretto
  su ISTA e UD.  llama ISTA 14.37 (prima richiesta) - 17.07 (warm).
  Estensioni finali: Q5_K aggiunto al dispatch MMVQ (head ISTA Q5_K:
  2.9 -> 2.4 ms, il residuo e' readback/sampling host) e path 4-wide
  BF16 in `dot<30>` per i GEMV hyper-connection (nessun guadagno: erano
  gia' bandwidth-bound a ~330 GB/s).  Run warm finali ISTA:
  **15.84 / 15.98 t/s** (llama warm 17.07, prima richiesta 14.37) ->
  ~93%% della warm, sopra la prima; 391 corretto, suite verde.
  Leva residua: CPU-MoE 42-48 ms (VNNI per IQ2_XXS/IQ3_S/Q2_0, atteso
  ~2-4 ms diluito dalla parallelizzazione), hc BF16 (~3.8 ms, vicino al
  limite di banda), MTP resta parcheggiato.
- 19 set VNNI CPU per IQ2_XXS/IQ3_S TENTATO e RIMOSSO (esperimento
  negativo): con la maschera di segno a 64 bit (packing diretto degli 8
  sign byte; per IQ2_XXS via `ksigns_iq2xs`) e `vpdpwssd` i kernel VNNI
  restano neutri o peggiori - IQ2_XXS 3.22 GB/s vs 3.40 maddubs; IQ3_S
  3.05 (set_epi32) / 2.92 (gather) vs 4.17 maddubs.  Il costo e'
  l'assemblaggio delle grid (16 load sparsi per 64 valori), che VNNI non
  ammortizza come su IQ2_S (grid = 4 qword per 32 valori).  Dispatch
  ripristinato ai maddubs; il VNNI resta solo per IQ2_S.  Stessa
  lezione dei value4 CPU: quando il kernel e' gia' al limite di banda
  parallela (MoE ~42-48 ms per 838 MB), il guadagno di ISA non paga.
- 19 set PREFILL CPU-MoE BATCHED (T>1, il gap grosso): il prefill era
  99.8% MoE GPU - ogni riga expert streamata per token via PCIe, 313 s
  per 1739 token (5.5 t/s).  Nuovo `qwen4_graph_moe_cpu_prefill`: readback
  di selected (T x NS) + mixed, counting sort in liste per esperto,
  mid/down a righe col token loop dentro, cosi' ogni riga di peso e'
  letta UNA volta per chunk e riusata per tutti i token che hanno scelto
  quell'esperto; stessi kernel Q8_K/fp32 del decode (numerica coerente).
  `DS4_QWEN4_CPU_MOE_PREFILL=0` per l'A/B.  Risultati su prompt 1739 tok:
  UD-IQ3_XXS **5.5 -> 24.3 t/s** (MoE CPU 65.7 s/chunk: mid 41 + down 25,
  readback 1.3 ms/layer, write 13 ms/chunk); ISTA **23.1 t/s**.  Riparato
  anche un bug pre-esistente esposto dal test: il down shexp IQ4_NL a
  T>32 falliva in `dense_blas` (tipo 20/42 non unpackati) - aggiunti al
  unpack e fallback a `matrix_dispatch`.  391 corretto su UD e ISTA col
  prefill CPU; suite verde.  Collo residuo: i kernel CPU pagano
  l'assembly della riga una volta per token (un kernel batched con B
  accumulatori la amortizzerebbe su B_e~34: stima 3-4x) e i ~45 GB di
  esperti per chunk contro ~42 GB di page cache (I/O SSD; chunk piu'
  grandi non aiutano il compute ma amortizzano i byte).
- 19 set KERNEL BATCHED per il mid prefill: `qwen4_cpu_dot_iq2_s_q8k_vnni_batch`
  (una riga di peso contro 8 attivazioni: grid/segni/sub-scale costruiti una
  volta per blocco e riusati, per token restano load q8 + sign flip +
  maddubs + dpwssd; accumulo per token nello stesso ordine del kernel
  singolo -> bit-identico) + helper `qwen4_cpu_row_dot_q8k_batch` (B=8, gli
  altri tipi restano per-token).  UD-IQ3_XXS: mid **853 -> 457 ms/layer**
  (1.87x), prefill **24.3 -> 31.0 t/s** (dal 5.5 iniziale: 5.6x); 391
  corretto.  Il down ora e' la meta' piu' grossa (658 ms/layer, fp32
  IQ4_NL per-token) e ISTA non beneficia (i suoi mid sono
  IQ2_XXS/IQ2_XS/IQ3_XXS: servono i gemelli batched per tipo, stesso schema).
  Prossime leve prefill, in ordine: (1) batched per IQ2_XXS/IQ2_XS/
  IQ3_XXS (ISTA); (2) batched fp32 per IQ4_NL/Q2_0 (down); (3) eventuale
  prefetch/pinning dei 45 GB di esperti (I/O).

### Handoff prefill (per chi riprende, nessuna ricostruzione necessaria)

Punto di partenza esatto:
- kernel batched modello (da copiare): `qwen4_cpu_dot_iq2_s_q8k_vnni_batch`
  (in `ds4.c`, sotto `qwen4_cpu_dot_iq2_s_q8k_vnni`, dentro il guard
  `DS4_CPU_IQ2S_Q8K_VNNI`); ogni kernel fa: costruzione grid/maschera/sub-scale
  UNA volta per blocco, poi `for b in 0..n-1 { load q8_b; maddubs; accumula
  in acc[b]; }`, fold `d * ys[b][i].d * reduce(acc[b])` per token a fine
  blocco.  Cosi' l'ordine per token e' quello del kernel singolo ->
  bit-identico (391 lo conferma).
- helper da estendere: `qwen4_cpu_row_dot_q8k_batch` (switch per tipo,
  oggi solo `DS4_TENSOR_IQ2_S`; gli altri tipi cadono nel loop per-token) -
  basta aggiungere i case dei nuovi twin.
- chiamante: worker `qwen4_cpu_moe_pf_mid_rows` (blocco `for base += 8`),
  gia' scritto per liste di 8; e' l'unico posto da toccare per il mid.
  Per il down: `qwen4_cpu_moe_pf_down_rows` (usa `qwen4_cpu_row_dot`,
  fp32) - li' serve il twin batched fp32 con B accumulatori
  (la parte peso-derivata e' lo shuffle LUT, ~25% del lavoro: guadagno
  atteso minore del mid).
- Ordine di valore su ISTA (byte di mid per tipo): IQ3_S 152 MB >
  IQ2_XS 81 > IQ2_S 84 (fatto) > IQ2_XXS 67 > IQ3_XXS 49; su UD il mid e'
  IQ2_S (fatto) + IQ3_S 1 layer, quindi il prossimo guadagno UD viene dal
  down fp32 IQ4_NL (658 ms/layer).
- Verifica: `make test-qwen4-cpu-dot` (worst rel invariati), poi probe
  aritmetico `391` e misura `DS4_QWEN4_MOE_PROFILE=1` + prompt 1739 token
  (`/tmp/prefill_prompt.txt`), confrontando mid/down ms per layer.
- Baseline attuale da battere: UD prefill 31.0 t/s (mid 457 + down 658
  ms/layer), ISTA 23.1 t/s.

### Aggiornamento: twin batched per tutti i tipi mid (fine campagna prefill)

Fatti, con lo stesso schema di `qwen4_cpu_dot_iq2_s_q8k_vnni_batch`
(grid + maschera segni + scale costruite una volta per step e riusate per
B=8 token; per token restano load q8, sign flip, maddubs/madd; ordine per
token identico al singolo -> bit-identico):

| twin | file/anello |
|---|---|
| `qwen4_cpu_dot_iq2_s_q8k_vnni_batch` | ds4.c, guard `DS4_CPU_IQ2S_Q8K_VNNI` |
| `qwen4_cpu_dot_iq2_xxs_q8k_batch` | ds4.c, guard `DS4_CPU_IQ2S_Q8K` |
| `qwen4_cpu_dot_iq2_xs_q8k_batch` | ds4.c (tabelle sign-shuffle locali) |
| `qwen4_cpu_dot_iq3_s_q8k_batch` | ds4.c (attenzione: return senza 0.125) |
| `qwen4_cpu_dot_iq3_xxs_q8k_batch` | ds4.c (fattore 0.25) |

Tutti agganciati in `qwen4_cpu_row_dot_q8k_batch` (switch per tipo); il
chiamante e' il solo loop `base += 8` di `qwen4_cpu_moe_pf_mid_rows`.

Numeri ISTA (prompt 1739 token, stesse condizioni):
- mid: 1003.6 -> **506.6 ms/layer** (2.0x) passando da IQ2_XXS a IQ2_XXS+
  IQ2_XS+IQ3_S+IQ3_XXS (i twin si sommano: 1003->904->828->637->507);
- prefill: **23.10 -> 33.13 t/s**; 391 corretto dopo ogni twin; UD
  invariato (il suo mid e' gia' tutto IQ2_S batched) e 391 ok.

Fase piu' grossa ora: il **down** (544 ms/layer ISTA, 658 su UD): usa
`qwen4_cpu_row_dot` fp32 per IQ4_NL/Q2_0, per-token.  Li' la parte
peso-derivata e' lo shuffle LUT (~25% del lavoro), quindi il twin batched
fp32 (B accumulatori, LUT condivisa) va scritto ma rende meno del mid.
Dopo quello restano solo I/O (45 GB di esperti/chunk vs 42 GB di page
cache) e il prefill a chunk piu' grandi per amortizzare i byte/token.

### Loop swap nel down (fatto, chiude la campagna)

`qwen4_cpu_moe_pf_down_rows` ora itera **token esterno / righe interne**
(un token per volta attraversa le righe dell'esperto): il vettore mid da
640 float resta in cache invece di essere riletto per ogni riga (prima
una volta per coppia (riga,token) = ~34x per riga).  Nessun cambio di
kernel, nessun cambio di numerica (ogni elemento usa lo stesso dot con
gli stessi input).

Risultati finali prefill (prompt 1739 token, stesse condizioni):

| | inizio sessione | fine |
|---|---|---|
| UD-IQ3_XXS | 5.5 t/s | **36.5 t/s** (6.6x) |
| ISTA | falliva | **35.2 t/s** |

Fasi UD: mid 397 + down 541 ms/layer (da 853 + 658); ISTA: mid 507 +
down 473 (da 1004 + 592).  `391` corretto su entrambi dopo ogni passo;
suite completa e `ds4-server` verdi.

Cosa resta (in ordine di valore):
1. **twin batched fp32 per il down** IQ4_NL/Q2_0 (`qwen4_cpu_dot_iq4_nl_simd`,
   `qwen4_cpu_dot_q2_0_simd`): B accumulatori + LUT dequant condivisa;
   leva maggiore ora che il down e' la fase piu' grossa su UD (541 ms);
2. **I/O**: ~45 GB di esperti per chunk contro ~42 GB di page cache;
   chunk piu' grandi amortizzano i byte/token (a T=8192 il peso dei byte
   per token cala di ~4.7x) ma non il compute;
3. il compute CPU aggregato e' ~60 GMAC/s su entrambe le fasi (3.8
   GMAC/s/thread): il limite e' il kernel, non piu' l'ordinamento.

### Twin batched fp32 per il down (fatto: chiude la campagna prefill)

`qwen4_cpu_dot_iq4_nl_batch` e `qwen4_cpu_dot_q2_0_batch` (guard
`__AVX512F__ && __AVX512BW__ && __AVX512VL__`, subito dopo i rispettivi
kernel singoli): la dequant del peso (LUT nibble IQ4_NL, unpack dei 4 campi
Q2_0) e' costruita una volta per blocco e riusata per 8 token; per token
restano solo i load di x e le FMA.  Numerica identica per token.

Il worker `qwen4_cpu_moe_pf_down_rows` e' passato a **token-block (8)
esterno / righe interne** via `qwen4_cpu_row_dot_fp32_batch_store`: il mid
dei token del blocco (8 x 2.5 KB) resta in L1 per tutte le righe.

Numeri finali (prompt 1739 token; stesse condizioni di tutte le misure):

| | inizio sessione | meta' | **fine** |
|---|---|---|---|
| UD-IQ3_XXS prefill | 5.5 t/s | 36.5 | **57.6 t/s** |
| ISTA prefill | falliva | 35.2 | **44.5 t/s** |

Fasi finali (ms/layer):
- UD: mid 381 + down **196** (da 853 + 658);
- ISTA: mid 568 + down **203** (da 1004 + 592).

Suite completa verde e `ds4`/`ds4-server` compilano dopo ogni passo;
`391` corretto su UD e ISTA a ogni iterazione (bit-identicita' dei twin
verificata a ogni passo, non solo alla fine).

Restano, in ordine: (a) il **mid** e' di nuovo la fase dominante (568 ms
ISTA): i twin esistono gia' per tutti e 5 i tipi, il margine e' nel
batch piu' profondo (provato B=8; B piu' alto = piu' accumuli in registro)
o nel tiling di xq; (b) **I/O/chunk**: ~45 GB di esperti per chunk contro
~42 GB di page cache - con `--prefill-chunk` piu' grande i byte per token
calano (a T=8192 ~4.7x) senza toccare il codice; (c) il compute aggregato
resta ~60 GMAC/s su entrambe le fasi (3.8 GMAC/s/thread).

### Esperimenti negativi (misurati e rimossi)

- **Accumulatore vettoriale in `qwen4_cpu_dot_iq2_s_q8k_vnni_batch`**
  (`accf[b] = fmadd(set1(dr*y.d), cvt(acc[b]), accf[b])` con una sola
  riduzione a fine riga, la forma di ggml): UD mid 381 -> 401 ms/layer,
  prefill 57.6 -> 53.5 t/s.  **Piu' lento**, oltre che non bit-identico:
  la riduzione orizzontale per blocco NON era il collo.  Rimosso.
- **Mid a token-block esterno / righe interne** (stesso schema che sul down
  aveva dato 2.8x): su UD mid 381 -> 387 ms/layer = rumore.  Rimosso,
  ripristinato il row-major: il mid non e' limitato ne' dal riuso di xq ne'
  dal pattern di scrittura.
- (precedenti, stessa sessione) VNNI per IQ2_XXS/IQ3_S: negativo; CUDA
  graph/dispatch: refutato dalle misure.

### Varianza delle misure (importante per chi confronta)

`DS4_QWEN4_MOE_PROFILE=1` stampa **6 volte per pass** (ogni 8 layer) e il
mid varia 420 -> 553 ms/layer **dentro la stessa run** su UD (mix di
esperti e stato della page cache).  Quindi:
- confrontare sempre **medie dei 6 print**, non l'ultimo;
- il prefill totale varia ~51-58 t/s (UD) e ~43-45 t/s (ISTA) tra run
  con lo stesso binario: differenze < ~10% non sono regressioni.
- Medie dell'ultima run UD: mid ~440, down ~225 ms/layer; ISTA: mid ~540,
  down ~196.

### Mid del prefill: il DNA e' il TRAFFICO delle attivazioni (misurato)

Bench isolato del twin battezzato, aggiunto al test harness
(`DS4_BENCH_DOT=1 ./tests/test_qwen4_cpu_dot`, funzione `time_batch_q8k`):
una riga di peso contro 8 attivazioni per chiamata, con 8 buffer (stesso set
ogni volta) o 4096 buffer (12 MB, set che scorre).

| IQ2_S, per thread | GMAC/s | GB/s di q8 |
|---|---|---|
| xq L1-hot (nbuf=8) | **50.1** | 57.2 |
| xq streamed (nbuf=4096, 12 MB) | **9.75** | 11.1 |
| kernel singolo token (riferimento) | 14.9 | — |
| IQ3_S L1-hot / streamed | 25.6 / 7.5 | 29.2 / 8.6 |

### Kernel a 2 righe: fatto, misurato e validato (non ancora integrato)

`qwen4_cpu_dot_iq2_s_q8k_vnni_batch2` (2 righe x 4 token: gli 8 accumulatori
stanno nei 32 zmm senza spill) riusa i load di q8 su due righe.  Misura nello
stesso run del bench:

| IQ2_S | L1-hot | streamed (12 MB) |
|---|---|---|
| batch x8 (1 riga) | 32.90 GMAC/s | 10.75 GMAC/s (12.3 GB/s q8) |
| **batch2 x4 (2 righe)** | 33.38 | **13.76** (7.8 GB/s q8) = **+28%** |

A parita' di MAC legge meta' byte di attivazioni: l'ipotesi del traffico
e' confermata.  **Correttezza**: `check_batch2()` nel test harness confronta
i 8 valori (2 righe x 4 token) col kernel a token singolo -> differenza
**0.00e+00**, bit-identico (l'ordine di accumulo per (riga,token) e' lo
stesso).  Il test resta verde.

Nota importante per l'integrazione: con 2 righe x 4 token l'assembly della
griglia e' per (riga, passo) -> 2 assembly per 4 token contro 1 per 8 token.
La forma giusta e' 2 righe x **8** token: nel bench fa 38.07 GMAC/s a caldo
(vs 26.19 del 2x4 e 49.56 del batch x8 a 1 riga) e 13.37 streamed = +17%
sul batch x8.  `check_batch2()` verifica 2x4 **e** 2x8, entrambi bit-identici
(0.00e+00).

### Mappa del prefill: il CPU-MoE e' il 93% del tempo (misurato)

`DS4_QWEN4_MOE_COUNT=1` stampa il lavoro vero di un chunk MoE:

```
MoE count T=1739 ne=512 ns=10 k_in=2560 k_ff=640 pairs=17390 MACs=56.98 GMAC act=64961 MB w=432 MB
```

- il profilo stampa ogni 8 chunk (`s_chunks % 8u == 0`) e ne compaiono 6 -> **~48 chunk
  per pass**, ognuno sull'intero prompt: 48 x 650 ms = **~31 s dei 33 s del prefill**;
- per chunk: **mid ~62%, down ~30%**;
- **il mid e' ~1:1 sul cammino critico**: `DS4_QWEN4_MID_DUP=1` (stesso risultato,
  kernel eseguiti due volte) porta il totale/chunk da 655 a 942 ms e il prefill da
  **52,4 a 37,0 t/s** -> +291 ms di mid = +287 ms di totale.  Ogni ms risparmiato nel
  mid e' un ms di prefill;
- **65 GB di attivazioni rilette per layer** contro 432 MB di pesi (rapporto 150:1):
  ogni riga di gate (e di up) ripercorre i q8 di tutti i token del suo esperto.

### Chunk-esterno nel mid: +5,2% end-to-end (tenuto, default)

Il loop del mid era **riga-esterno / token-interno**: ogni riga rifaceva il walk di
~100 KB di attivazioni (~34 token x 2,9 KB) contro una L1 di 32 KB, quindi riuso
tra righe consecutive oltre la L1.  Con il **token-chunk esterno (8 token = 23 KB,
L1-hot) e le righe interne** (stessa trasformazione che nel down diede +2,8x):

| | mid ms | total ms | prefill |
|---|---|---|---|
| riga-esterno | 417-426 | 643-658 | 52,18 t/s |
| **chunk-esterno** | **393-401** | **617-628** | **54,91 t/s** |

Aritmetica identica per (riga, token) e stesso ordine di accumulo -> **bit-identico**
(391 ok, test verdi).  Default attivo; `DS4_QWEN4_MID_ROWOUTER=1` ripristina il
vecchio ordine per A/B (e' li' che valgono NOKERNEL/DUP).

### Ipotesi scartate oggi (con i numeri)

| ipotesi | misura | esito |
|---|---|---|
| gli store sparsi in `mid` sfrattano la L1 | bench +mid-stores: 50,29 -> 46,69 GMAC/s | ~7%, **non e' il collo**; rimosso |
| il mid e' limitato dai byte di attivazioni | kernel a 2 righe: 2,3x nel bench streamed, **0 nel modello** | **refutata** |
| SMT (16 vs 8 thread) | mid 456 vs 434 ms, aggregato piatto | collo condiviso, non per-core |
| "16 token per chiamata e' 2x" | **artefatto mio**: i kernel chiudono a `n=8`
  (`if (n > 8u) n = 8u;`) e il printf divideva per 16 | da rifare sul serio; bench ora
  clampa a 8 per non ripetere l'errore |

### Kernel a 16 token: misurato (+12% streamed), non integrato (con motivo)

L'ipotesi era latenza/MLP (con 8 token solo 8 catene di accumulo indipendenti).
Il kernel IQ2_S e' stato portato a `n <= 16`: il load q8 era **gia'** dentro il
loop dei token (una sola variabile viva), quindi servono solo ~6 zmm in piu' e
ci stanno nei 32.  Misure nello stesso run:

| | L1-hot | streamed (12 MB) |
|---|---|---|
| batch x8 | 43,88 GMAC/s | 12,40 GMAC/s |
| **batch x16** | 43,43 (pari) | **13,89 (+12%)** |

Correttezza: `check_batch2()` verifica ora anche x16 contro il kernel a token
singolo -> **0,00e+00** (bit-identico); test verde.

**Non integrato nel worker**, e il motivo e' misurato: con il chunk-esterno (che
vale +5,2%) un chunk da 16 token = **46 KB di attivazioni, oltre la L1 da 32 KB**,
quindi si perderebbe la localita' appena guadagnata.  Il kernel a 16 resta come
materiale per un design a righe bloccate con piu' token tenuti in registri.

Trappola da non ripetere: il "+2x con 16 token" che avevo visto era un
**artefatto** (i kernel chiudono a `n = 8` e il bench divideva per 16).  Ora il
bench ha un clamp per tipo (`maxn`: 16 per IQ2_S, 8 per gli altri): **se un
GMAC/s sembra troppo bello, controllare il clamp**.

## Il collo residuo e' lo SCALING, non il kernel (misurato)

Sintomo di partenza: il modello fa **8,9 GMAC/s per thread**, il bench *streamed*
(che va in DRAM) ne fa 12, il bench L1-hot 50.  Il modello era quindi **piu' lento
di un microbenchmark in DRAM**: non poteva essere banda ne' latenza.

Ho aggiunto al bench `time_model_like`, che riproduce il pattern del modello
(lista di ~34 token **sparsi** in uno scratch da T=1739 vettori, una **riga di
peso nuova a ogni chiamata**, chunk da 8) e ho aperto le due variabili:

| variante | GMAC/s |
|---|---|
| dense, same-row, no-mid | 32,76 |
| **scattered, fresh-row, no-mid** (= modello) | **31,99** |
| dense, same-row, +mid 35,6 MB | 31,95 |
| **scattered, fresh-row, +mid 35,6 MB** (= modello) | **31,31** |

Conclusioni (tutte **negative**, utili):

- **i puntatori sparsi non costano nulla** (32,8 -> 32,0);
- **cambiare riga a ogni chiamata non costa nulla** (32,8 -> 31,7);
- **il buffer `mid` da 35,6 MB scritto nel pattern del modello costa il 4%**
  (31,99 -> 31,31): non e' lui a sfrattare le attivazioni.

Quindi il pattern del modello, **a un thread, gira a ~31 GMAC/s**, cioe' 3,5x il
rate per-thread del modello.  Il divario e' **parallelo**: 31,3 GMAC/s x 8 core
= 250 GMAC/s ideali contro i **142 aggregati** misurati nel modello = **56% di
efficienza di scaling**.  Con scaling ideale il mid scenderebbe da ~400 a ~230
ms e il prefill andrebbe a **~70 t/s**.

Conseguenza per il lavoro futuro: **smettere di ottimizzare il kernel a un
thread** (e' gia' a 31/50 GMAC/s e le sue varianti non trasferiscono) e attaccare
lo scaling.  Esperimenti pronti, in ordine:

1. **bench multi-thread dello stesso `time_model_like`** (N thread, GMAC/s
   aggregati) per la curva di scaling 1/2/4/8/16 thread: dice se il muro e' la
   L2 condivisa da SMT (2 thread per core, ognuno con ~1 MB di pesi streamati da
   un esperto contro 1 MB di L2 -> thrash) o la banda aggregata L3/DRAM.
2. **Affinity** (`taskset`): 16 thread su core fisici contro sparpagliati su
   CCD diversi -> se la L3 per-CCD conta.
3. Se il muro e' la L2 condivisa: **tiling a blocchi di righe** (~32 righe =
   52 KB di pesi) dentro il chunk, cosi' il working set per thread scende sotto
   la L2 privata, pagando piu' riletture delle attivazioni (che pero' stanno in
   L2/L1); oppure **fusione mid+down** per chunk (il `mid` locale di 8 token x
   640 = 20 KB sta in L1) per non far passare 35,6 MB di scritture per le cache.

### Curva di scaling: il tetto della macchina e' ~165 GMAC/s aggregati

`time_model_like_mt` (stesso pattern, N thread concorrenti):

| thread | GMAC/s aggregati | per thread |
|---|---|---|
| 1 | 30,6 | 30,6 |
| 2 | 60,6 | 30,3 |
| 4 | 104,2 | 26,0 |
| 8 | **163,6** | 20,5 |
| 16 | 167,4 | 10,5 |

- da **8 a 16 thread non si guadagna nulla** (167 vs 164) e il rate per thread si
  dimezza: SMT non serve (coerente con la misura nel modello, 8 vs 16 thread);
- il **modello fa 142 GMAC/s aggregati = 86% di questo tetto misurato**;
- il tetto e' di **banda condivisa**, non di kernel: per core la coppia SMT streama
  2 x 525 KB di pesi (le k_ff righe di un esperto) e deve tenere 2 x 100 KB di
  attivazioni = **1,25 MB contro 1 MB di L2** -> le attivazioni finiscono in L3, e
  il muro e' la banda L3 aggregata (~190 GB/s, che e' anche il numero che esce
  dai byte: 163 GMAC/s x 1,14 B/MAC).

**Conseguenza: il mid e' quasi al limite fisico di questa macchina per questo
pattern.**  Il margine residuo e' 1,17x (arrivare da 142 a 165).  Per andare
oltre bisogna **ridurre i byte per MAC**, cioe' aumentare il riuso
(attivazioni rilette 1270 volte per i pesi che le accompagnano).

Esperimenti pronti, in ordine di rapporto resa/costo:

1. **Prefetch NTA dei pesi** (`_mm_prefetch(..., _MM_HINT_NTA)` sulle righe di
   peso prima del kernel): NTA mette la linea in L1 con priorita' di sfratto in
   L2/L3, quindi il flusso dei pesi **non alloca in L2** e le attivazioni ci
   restano -> si smette di pagare la L3.  Poche righe, attacca il meccanismo
   appena quantificato.
2. **Ritestare il kernel a 2 righe nel modello** (e' ancora in albero, validato
   bit-identico e usato solo dal bench): dimezza i byte di attivazioni per MAC,
   cioe' dimezza il traffico verso la L3.  Il test precedente (0 guadagno) era
   **prima** del chunk-esterno, quando la L1 era il problema; ora il problema e'
   la L2/L3 condivisa, quindi la conclusione va rifatta con i numeri di oggi.
3. Se 1+2 non bastano: **fusione mid+down per chunk** (il `mid` di 8 token x 640
   resta in L1) per non far passare 35,6 MB di scritture per le cache, oppure
   **blocchi di righe** (~32 righe = 52 KB di pesi) per tenere il working set per
   thread sotto la L2 privata.

### Prefetch NTA dei pesi: alza il tetto del bench (+10%), in-modello e' nel rumore

Implementato (2 righe nel kernel IQ2_S: `_mm_prefetch(row + off, _MM_HINT_NTA)`
su tutte le linee della riga di peso).  Effetto misurato:

| thread | senza NTA | con NTA |
|---|---|---|
| 1 | 30,6 | 31,7 (+4%) |
| 2 | 60,6 | 63,1 (+4%) |
| 4 | 104,2 | 112,2 (+8%) |
| 8 | 163,6 | **180,2 (+10%)** |
| 16 | 167,4 | 178,4 (+7%) |

Il tetto aggregato passa da ~165 a **~180 GMAC/s**: il meccanismo (pesi che
sfrattano le attivazioni dalla L2 condivisa) e' confermato a livello di microbench.

In-modello (UD, stesso prompt): **mid 397 -> 385-394 ms (-2,5%)**, down 207 ->
197-202 (-4%), `391` corretto, ma **prefill 55,01 vs 55,60 t/s**: dentro il rumore
di un singolo run (la varianza run-to-run di questo path e' +-5%).  Tenuto perche'
non costa nulla e le fasi migliorano; la conferma end-to-end richiede un A/B a
3 run che non ho fatto (dichiarato, non nascosto).

Lettura: il modello sta a 142 GMAC/s aggregati contro un tetto di 180, cioe' al
**79%**: il margine residuo e' ~1,27x e va cercato riducendo i **byte per MAC**
(2 righe: dimezza le attivazioni; fusione mid+down: toglie 35,6 MB/chunk di
scritture dalle cache), non ottimizzando il kernel a un thread.

### Tetti misurati di ENTRAMBE le fasi: il CPU-MoE e' al 76-79% del limite

Curve di scaling multi-thread dello stesso pattern (file di test):

| thread | mid 1 riga | mid 2 righe | down IQ4_NL |
|---|---|---|---|
| 1 | 31,3 | 34,3 | 34,4 |
| 2 | 61,9 | 67,4 | 64,8 |
| 4 | 119,2 | 131,1 | 117,0 |
| 8 | 176,5 | **184,7** | **187,1** |
| 16 | 165,9 | **185,6** | 184,6 |

- **down: tetto ~187 GMAC/s, il modello ne fa 143 = 76%**;
- **mid: tetto ~176-186, il modello ne fa 142 = 79%**;
- **il kernel a 2 righe alza il tetto solo del 6%** (176 -> 185) pur dimezzando i
  byte di attivazioni per MAC: **il muro non e' la banda**, e' qualcos'altro
  (latenza/throughput di richieste, non byte).  Per questo il test a 2 righe era
  risultato neutro in-modello: non c'era margine da prendere;
- da 8 a 16 thread il tetto del mid **cala** (176 -> 166 a 1 riga): SMT non solo
  non serve, in questo pattern **ruba** throughput.

Conseguenza: il CPU-MoE e' vicino al limite di questo design sulla macchina.
Margine residuo realistico **~1,2x** (mid 400 -> ~340 ms, down 200 -> ~180),
cioe' prefill **55 -> ~63-65 t/s** al massimo, e il 20% mancante e' overhead del
modello (silu ~4,5% misurato, store, walk delle liste, indirezione `tok_idx`),
non kernel.

### Prefill sulla GPU: 5,41 t/s (contro 55 del CPU-MoE) -> strada chiusa

`DS4_QWEN4_CPU_MOE_PREFILL=0`, stesso prompt: **5,41 t/s**.  Lo streaming dei pesi
degli esperti verso la VRAM domina; il CPU-MoE e' 10x meglio.  Non e' una
alternativa da riprendere.

### Bilancio finale di oggi (UD)

| | prefill |
|---|---|
| inizio campagna | 5,5 t/s |
| prima di oggi | 52,2-54,0 t/s |
| **dopo il chunk-esterno** | **55,60 t/s** |

`391` corretto su UD e ISTA con il nuovo default, build 0 errori, test verdi.

### Il kernel a 2 righe nel modello: 0 guadagno (esperimento negativo)

A/B nello stesso binario, `DS4_QWEN4_MID_R2=0|1`, stesso prompt e cache:
mid **451.1 / 452.6 ms/layer** = identici.  Il +17% del bench **non si
trasferisce**: in-situ contano cose che il bench non ha (l'assembly doppio per
token, il walk della lista token, le scritture in `mid`, l'overhead di
chiamata), quindi il mid **non e' limitato solo dal traffico di attivazioni**.
Il branch del worker e' stato rimosso (0 residui), il kernel a 2 righe e il
suo check restano nel test come strumento di misura (non usati in
produzione).

Lezione operativa: il bench isolato serve a scartare in fretta, **non** a
promettere guadagni; qualunque variante va validata in-modello subito dopo.
Prossime ipotesi, in ordine: (1) il mid a **B=8 con 2 righe** ma chunk di 8
*senza* doppio assembly (cioe' assembly condivisa fra le due righe: richiede
di estrarre grid+segni una volta e applicarli a due accumulatori, possibile
solo se la riga non entra nel calcolo della maschera... non e' il caso di
IQ2_S); (2) le **scritture** in `mid` (8 store sparsi da 4 B per riga-chunk)
o il walk delle liste; (3) accettare il plateau e passare ad altro.

Il mid del modello misura ~9.4 GMAC/s per thread (150 GMAC/s aggregati su
16 thread): **coincide con il caso streamed**, non con il tetto a caldo.
Quindi il collo non e' l'aritmetica del kernel (a caldo fa 3.4x il kernel
singolo) ma il **traffico della stream di attivazioni**: ~11 GB/s per
thread, ~180 GB/s aggregati, che e' il limite L3/uncore di questa
piattaforma.  Il set per esperto e' piccolo (~34 token x 2.9 KB = 100 KB)
ma le righe di peso che scorrono (820 B ciascuna, una volta sola) lo
sfrattano di continuo dalla cache.

**Conseguenza: la riscrittura a registri bloccati e' giustificata e il suo
tetto e' misurabile.**

Stato: ~2.2 MAC/cycle/thread (0.07 `vpdpwssd`/cycle) contro i 32 teorici.
Esperimenti fatti e scartati, tutti misurati nello stesso binario:
1. accumulatore vettoriale al posto della riduzione per blocco -> piu' lento;
2. token-block esterno / righe interne (riordino di loop) -> neutro;
3. batch a 16 token (`DS4_QWEN4_MID_B16`) -> piu' lento (368.6 -> 381.9);
4. VNNI al posto di maddubs per IQ2_XXS/IQ3_S -> negativo;
5. silu: costa 18.5 ms/layer (4.5%), la versione a 8 vie ne recupera ~8.
Quindi non e' il conteggio delle operazioni, non e' la silu, e non si
risolve aumentando le catene di load in volo.

Bilancio di registri del kernel attuale (twin VNNI, chunk di 8 token, 10
blocchi): per (riga, chunk) si fanno 80 load di q8 (8 token x 10 blocchi,
~23 KB) e ~14 op di assembly per blocco; il calcolo puro e' ~115 cicli ma
il tempo reale e' ~500 cicli per (riga, chunk).  Il traffico di xq e' ~32 GB
per layer (1.39M coppie (riga,chunk) x 23 KB): e' il candidato principale,
perche' non e' ne' banda (73 GB/s aggregati) ne' latenza (B=16 peggiora).

Forma suggerita (non provata): tenere il q8 di un chunk di token in
registri e riusarlo su PIU' righe.  Un blocco di 8 token occupa 4 zmm
(8 x 32 B); ogni riga pero' richiede 8 accumulatori (uno per token) che
devono sopravvivere ai 10 blocchi, quindi con 32 zmm il massimo e' 2-3
righe per volta + temporanei: il guadagno atteso e' 2-3x sul traffico di
xq, non 8x.  L'alternativa (righe nelle corsie SIMD) richiede gather
per riga ed e' probabilmente peggiore.

Primo passo consigliato in una sessione nuova: **non** scrivere subito il
kernel, ma un bench isolato del twin battezzando 2 casi (xq in L1 vs xq
freddo) per stabilire il tetto del kernel attuale a caldo; senza profiler
(`perf`/`ncu` non utilizzabili qui) e' l'unico modo per sapere se il
guadagno e' nel kernel o nell'ambiente di memoria del modello.

- **`silu()` costa 18.5 ms/layer, ~4.5% del mid** (misurato nello STESSO
  binario con `DS4_QWEN4_MID_SILU=0|1|2` = silu scalare / vettoriale / niente:
  mid 412.9 / 404.8 / 394.4 ms per layer, stesso prompt, stessa cache).  La
  stima precedente di ~45 ms era rumore tra run: **misurare sempre nello
  stesso binario con uno switch**, non tra build diverse.  La versione
  vettoriale a 8 vie (`qwen4_cpu_silu8_mul`, exp con esponente clampato,
  ~1e-7 rel.) recupera ~8 ms/layer = 2% del mid, ~1.2% del prefill: tenuta,
  e `391` corretto su UD e ISTA (la risposta cambia wording, resta corretta).
- Diagnostica: TIMING=4 = stage GPU; NO_MMVQ=1 = A/B dense.
- 19 set VERIFICA vs llama (log reali, non stime): riferimento UD =
  21.85 tok/s (45.77 ms/token, 200 tok dopo prefill 30.5K a 177 t/s,
  7 thread, auto-fit: /tmp/llama-ref.log); run 17 set a 128K 14.89 tok/s
  steady su 44K token. ISTA: 14.37-17 tok/s (/tmp/qwen38-ista-32k.log).
  ATTENZIONE: -ncmoe/-ot espliciti disattivano l'auto-fit e fanno crollare
  llama a ~1.7 tok/s (misurato in llama-bench): per il riferimento usare
  il launcher ~/bin/qwen38-flashnext (auto-fit, mmap). ds4 oggi: UD 9.10
  t/s q8k (A/B 7.05 fp32, macchina rumorosa), ISTA 4.11 t/s su 8 token.
  ISTA CPU-MoE copre 10/48 layer (gate/up IQ2_XXS x9, IQ2_XS x10,
  IQ3_XXS x6 fuori allow-list; 30 down Q2_0) -> 38 layer in streaming
  GPU. Per raggiungere llama su ISTA servono kernel CPU per
  IQ2_XXS/IQ2_XS/IQ3_XXS (gate/up) + Q2_0 (down) e allow-list estesa;
  su UD la distanza residua e' ~1.3-1.7x dopo il kernel int8.
- 18 set CAMPAGNA 32K (stadi): boot -c 32768 OK (VRAM 7.8GB). Prefill
  3681 tok = 592s = 6.2 tok/s; replica identica warm = 572s (NON fill:
  e compute!). Causa: niente tiled/mm per Unsloth streamed (gate
  has_mm: whole-tensor residency -> OOM/illegal; commento autorevole
  in codice). Chunk scaling lineare (609tok=96s) -> chunk grandi non
  aiutano. Stima 30K = 80min -> full-32K bloccato in-session
  (opzione overnight). Decode a 3.7K ctx = 4.4 tok/s (attention OK
  a 4K). TTFT warm 3.0s vs cold 8.0s (prompt 18tok). Zero P0 in
  ~7.5K tok odierni (tutti corretti). Prossimo: tiled streaming-aware
  (medio-grande) o accettare + overlap; 32K overnight su decisione.
- 18 set KERNEL v2 staged (smem, gate+up+silu, 4 righe/blocco): bench
  MATCH v1 (maxrel 2.6e-4, reorder noise) ma pari velocita (0.144 vs
  0.141ms: lo staging non aiuta lo zero-copy). Server: MoE -10%
  (4.55->4.11ms/layer), end-to-end invisibile, 391 TRUE, traiettoria
  identica. MA bug latente trovato: v2 skippava lo shared-slot (UD3
  ha shared!) -> output corretti solo per shared-gate~0. Gate stretto
  a st==UINT_MAX (bench/shared-less only); server torna v1-exatto.
  dmon: latency-bound confermato (64W, membw 4%, SM 97%). Divario
  bench-server 18x inspiegato (non clock/fill/scatter) -> bench solo
  A/B relativo. Server spento, VRAM baseline.
- 18 set B6-validazione corta: ab long-form impraticabile (run 600 tok
  collassa in !-loop: vittima P0, non numerica) -> batteria corta
  (6 problemi aritmetici, max 12 tok, P0-immune): UD4 6/6, ISTA 6/6,
  zero flat. Numerica validata su 3 pack; long-form resta gated da P0.
  ~1.0-1.15 tok/s (109 tok risposta + 500 tok thinking, flat). Riferimento
  llama: 18-19 tok/s -> divario ~18x, tutto paging on-demand (GPU idle).
  Ottimizzazioni (resident-dense, value4/tiled, prefetch) dopo la qualita.
  Probe 50x/300 liberi superati da ab-math; sanitizer non disponibile.
  Server spento, VRAM libera.
- 17 set B7-primo-run (DIRECT_MODEL): boot OK, 600 tok generati senza
  errori/Xid, decode ~1.4 tok/s (on-demand paging, GPU quasi idle). Troncato
  nel thinking (come il test 2 llama): 391 non ancora verificato. Server
  tenuto caldo su 8892 per il follow-up; page cache 39 GB, swap +3 GB.

- 17 set B8.1 (disegno sidecar): la testa esterna diventa il blocco 48
  mancante: shape 49/1 via `--mtp-model` (override dopo il trim),
  `layer[48]` bindato dal sidecar (attn+indexer+hc+MoE+nextn; embed/head
  restano del trunk per `shared-*`), resto della spec machinery invariato
  (`mtp_step`/verify/snapshot lavorano su (m,w) generici). Da toccare:
  gate `mtp_path` per qwen4, firma config (niente globali), ratios[48]=4
  sintetico, validazione 0..48, gate residency streaming-aware, MoE draft
  in streaming automatico. Prima `shared-Q8_0` (zero tipi nuovi).
- 17 set B8.2 (MTP funzionante): sidecar assorbito come shard extra
  (fix: MAP_FIXED tail dentro slack riservata; btrfs rifiuta NOREPLACE
  spurio; dangling ngram risolto). 41 cicli spec: first-accept 66%,
  24 double-accept. Velocita ~parita (draft head full-vocab 0.5 GB × N
  drafts domina). Prossima leva: draft head gathered per Q6_K (16 MB
  una tantum). Request-2 e teardown restano aperti.
- 18 set teardown SIGSEGV (CHIUSO): crash deterministico all'exit dopo
  engine_close (CLI e server). Trovato con print-bisect: (1) model_close
  munmap 80GB PRIMA di gpu_cleanup -> UVA/driver fault; fix: ordine
  invertito (cleanup, DeviceReset, poi unmap). (2) crash post-main negli
  exit handler (race thread driver vs teardown libc); fix: fflush+_exit
  in main CLI-batch e server (REPL invariato per terminal restore).
  CLI rc=0, server kill senza core, VRAM a baseline. TEMP rimossi.
- 18 set B8.3 (gathered draft head, opt-in): `DS4_QWEN4_MTP_DRAFT_VOCAB`
  + scoring Q6_K su subset (entry `matmul_weights_tensor`, chain usa lo
  stesso subset; test `test_qwen4_drafthead` verde). Breakdown misurato:
  verify 545-650 ms/riga (streaming trunk), draft block 1.3 ms, draft head
  full 145 ms; il sync mostra che il costo draft (~66 ms) e fetch MoE
  esperti via PCIe, non GEMV. Subset 7.5K/15K: acceptance 77-78% vs
  90% full-head (output byte-identici, A/B pulito) e tok/s in regressione
  (1.42 -> ~1.0): il subset perde piu acceptance di quanto risparmi.
  Default resta full head; leva vera = residency esperti trunk, non draft.
  mancante: shape 49/1 via `--mtp-model` (override dopo il trim),
  `layer[48]` bindato dal sidecar (attn+indexer+hc+MoE+nextn; embed/head
  restano del trunk per `shared-*`), resto della spec machinery invariato
  (`mtp_step`/verify/snapshot lavorano su (m,w) generici). Da toccare:
  gate `mtp_path` per qwen4, firma config (niente globali), ratios[48]=4
  sintetico, validazione 0..48, gate residency streaming-aware, MoE draft
  in streaming automatico. Prima `shared-Q8_0` (zero tipi nuovi).

## Sequenza proposta (opzione B)
1. Loader multi-shard + shape no-MTP (solo CPU/build, nessun run).
2. Tipi fase 1 in CPU-ref, kernel CUDA, PLE IQ4_NL on-read.
3. Boot ctx 32K + 391 su UD-IQ3_XXS (via esplicita singola).
4. A/B contro llama IQ3_XXS (stessi 3 probe + bench 400 tok).
5. Solo se i numeri tornano: ctx 64K/128K, KV q4_0, fasi 2-3 (UD-IQ4_XS, ISTA).

### Chunk da 16 e 2 righe al tetto (MT bench, stesso run)

| 16 thread | GMAC/s |
|---|---|
| chunk 8, 1 riga (config del modello) | 152,1 |
| chunk **16**, 1 riga | 161,8 (+6%) |
| chunk 8, **2 righe** | **169,0 (+11%)** |
| a 8 thread: 2 righe vs 1 | 164,0 vs 139,3 (**+18%**) |

In questo run il modello fa 142 = **93% del tetto riprodotto** per la sua
configurazione esatta (chunk 8, 1 riga, 16 thread).  Le due varianti alzano il
tetto del 6-18% a 16 thread; il prossimo passo e' integrarle misurando il **t/s**
end-to-end (non i ms di fase, come nell'errore precedente).

### Correzione importante: la VM e' una COSTANTE, la variabile e' l'IO di pagina

Misurato: `qemu-system-x86` e' su da 5 giorni con 4d11h di CPU = **88% di un core
di media**, e 5 campioni istantanei danno 88,4% ogni volta; 16,8 GB di RSS.
Era quindi **ugualmente attiva** durante i run da 55 t/s e durante quelli da
40-46: **la VM non spiega la differenza**.  (La nota che avevo scritto prima,
"la VM contamina le misure", era sbagliata: la VM e' parte del baseline della
macchina da giorni.)

La variabile vera e' la **page cache / IO di lettura pesi**:

- modello **78 GB** contro **62 GB di RAM** (22 usati, di cui 16,8 della VM),
  buff/cache 40 GB, swap 7 GB in uso;
- `fincore`: shard 2 (49,6 GB) ha **3,2 GB residenti**, shard 3 (32,4 GB) 28,7 GB
  -> una frazione grande del modello non e' in cache;
- i run consecutivi salgono in modo **monotono** (52,60 -> 53,33 -> 54,00 e
  44,76 -> 46,53): e' *warming* della cache, non rumore;
- **misurato con /proc/PID/io: 33 GB letti dal disco in UN pass di prefill**
  (45,04 t/s = 38,6 s; a ~3,5 GB/s sono ~9 s = **~24% del tempo speso in IO**).

Conseguenze:

1. il "76-79% del tetto di calcolo" misurato col bench **e' in gran parte IO di
   page fault sui pesi mmap**, non inefficienza dei kernel: il bench (che non
   tocca il file del modello) misura il calcolo puro;
2. spiega anche perche' saltare i kernel del mid rallentava il totale: l'attesa
   IO si spostava sulla fase down invece di essere nascosta;
3. **la leva piu' grossa rimasta non e' un kernel ma la RAM disponibile per la
   cache**: il working set MoE di un pass e' ~48 GB, la cache oggi arriva a ~40 GB
   (VM accesa).  Con la VM spenta o ridotta a 4 GB la cache sale a ~52 GB e il
   working set ci sta: il traffico disco per pass puo' scendere verso zero, con
   un guadagno atteso fino a ~1,25x (45 -> 55+ t/s, e ~65-68 su cache calda).

Metodo per misurare l'IO di un run (da riusare):

```sh
./ds4 ... > log 2>&1 & P=$!; LAST=0
while kill -0 $P 2>/dev/null; do V=$(awk '/read_bytes/{print $2}' /proc/$P/io); [ -n "$V" ] && LAST=$V; sleep 2; done
echo "$((LAST/1000000000)) GB letti dal disco"
```

Prima di ogni campagna: **controllare cache e load** (`fincore`, `uptime`) e
confrontare solo run nelle stesse condizioni.

### 2 righe nel worker: rimosso (crash), diagnosi scritta

Il kernel a 2 righe e' in albero e validato bit-identico dal test
(`check_batch2`), e nel bench MT alza il tetto del 6-18% a 16 thread.  La
versione integrata nel worker `qwen4_cpu_moe_pf_mid_rows` **crasha
deterministicamente** (segfault, anche con prompt piccolissimo).  Analisi con
gdb sulla chiamata di coppia:

```
n = 1 (coda dispari: chunk di 1 token)
ys[0] = 0x3786230d  -> puntatore q8 selvaggio
dr, xa, xb letti invece corretti -> la riga di peso e' buona, e' l'xq del
primo token a essere marcio
```

Quindi o `xq` non e' quello che si crede in quella posizione, o c'e' una
corruzione a monte della chiamata.  Da riprendere **solo** con un debug mirato
(print di base/lo/hi/m/tok_idx + assert sulle lunghezze prima della chiamata) e
non riprovando l'integrazione cosi' com'e'.  Il codice della variante e' in git
(`qwen4_cpu_dot_iq2_s_q8k_vnni_batch2`, ~ds4.c:6690) e il bench la misura.

### Read-ahead dei pesi (MADV_WILLNEED): misurato NEGATIVO, rimosso

Idea: se il 24% del prefill e' I/O su page fault, chiedere al kernel di leggere in
anticipo i tre tensori degli esperti del layer (~900 MB) dovrebbe farli arrivare
durante il calcolo invece che a richiesta.

| | prefill | disco letto | mid | down |
|---|---|---|---|---|
| senza hint | **54,88 t/s** | 37 GB | 402,0 | 205,7 |
| con MADV_WILLNEED | 52,47 t/s | **50 GB** | 420,1 | 217,6 |

Peggiora **entrambi** i lati: il read-ahead completo del layer sfratta pagine che
sarebbero state riusate, e il traffico disco sale del 35%.  E' il caso classico di
read-ahead contro un working set (~48 GB/pass) piu' grande della cache
disponibile (~40 GB con la VM accesa).  **La conclusione operativa e' che quei
~24% di I/O non si nascondono con una hint: servono piu' RAM (o meno byte).**

### CORREZIONE: l'IO non e' sul cammino critico (3 run controllati)

Test decisivo, 3 run consecutivi misurando **t/s e GB letti dal disco** insieme
(stesso codice, stesso prompt, macchina nelle stesse condizioni):

| run | prefill | disco letto |
|---|---|---|
| 1 | 55,63 t/s | 33 GB |
| 2 | **56,61 t/s** | **42 GB** |
| 3 | 56,43 t/s | 41 GB |

**Nessuna correlazione fra disco e t/s** (il run che legge di piu' e' il piu'
veloce).  Quindi:

- l'IO di page fault **e' sovrapposto al calcolo**: con 16 thread il thread che
  prende il fault non ferma gli altri, e i 33-42 GB/pass si nascondono;
- la stima "~24% del tempo in IO" (33 GB / 3,5 GB/s) era **ingenua** e va
  scartata: i byte si leggono, ma non sono sul cammino critico;
- i run a 40-46 t/s visti prima erano **contention transitoria** (i miei stessi
  run crashati, le sessioni gdb, la build), non disco e non la VM: la VM e'
  costante a 88% di un core da 5 giorni.

**Conseguenza (correzione della correzione): liberare RAM per la cache NON e' la
leva** che avevo indicato.  Il confronto giusto per il mid e' con il tetto del
bench **a 16 thread** (il modello usa 16), non con il picco a 8 thread:

| | GMAC/s |
|---|---|
| modello (mid, 16 thread) | 142 |
| bench MT 16 thread, chunk 8, 1 riga | 152-166 |

cioe' il modello e' all'**86-92% del tetto riprodotto nella sua stessa
configurazione**: il margine residuo e' ~10%, ed e' overhead del modello (silu
~4,5% misurato, store, walk delle liste), non kernel ne' disco.

### Bug trovato e corretto: silu8 scriveva 16 float su array da 8

`qwen4_cpu_silu8_mul` faceva `_mm512_loadu_ps` + `_mm512_storeu_ps` (64 byte = 16
float) mentre tutti i chiamanti dichiarano `float[8]`.  Conseguenze:

- **percorso a riga singola** (quello di produzione): i 32 byte in piu' finivano
  nell'array adiacente (`vb`), che non serviva piu' -> **UB latente ma innocuo**;
- **percorso a 2 righe**: l'oggetto adiacente e' l'array dei puntatori q8 ->
  `xq[0..3]` sovrascritti -> segfault con `xq[0] = 0x3786230d`.  Era **questa** la
  causa del crash che avevo attribuito a un bug di indicizzazione.

Correzione: maschera a 8 lane (`_mm512_maskz_loadu_ps` / `_mm512_mask_storeu_ps`),
che legge e scrive solo 32 byte e lascia identici i risultati delle 8 lane valide.
Build pulita, `391` corretto, test verdi.

### 2 righe nel worker, misurate DOPO il fix: NEGATIVE (e ora la misura e' pulita)

Senza piu' il crash, l'A/B nello stesso binario:

| | prefill | mid | down |
|---|---|---|---|
| R2=0 (default) | **56,09 t/s** | 389,7 | 202,9 |
| R2=1 (2 righe) | 51,09 t/s | **438,8** | 216,0 |

**+12,6% sul mid**, cioe' peggio: l'assembly della griglia e' per riga, quindi il
percorso a coppie la raddoppia per token, mentre i byte di attivazioni risparmiati
non erano il collo (coerente con ogni test di oggi).  **Branch rimosso**; il
kernel e il bench restano.

Lezione: il bench MT diceva +11% di tetto, il modello dice -12,6%.  **Il bench
misura il tetto di una configurazione, non prevede l'effetto end-to-end.**

### Store non temporali nel mid: misurati NEGATIVI, rimossi

Idea: il mid e' scritto con 4 byte per (riga, token) in un buffer da 35,6 MB, una
linea nuova per token, quindi uno store normale paga il read-for-ownership che
quello non temporale salta.

| | prefill | mid | down |
|---|---|---|---|
| NT=0 | **50,57 t/s** | 434,0 | 229,4 |
| NT=1 | 46,81 t/s | 476,9 | 242,1 |

**-7%**.  Motivo tecnico: sono 4 byte in linee diverse, cioe' il caso peggiore per
i write-combining buffer (flush di linee parziali), e in piu' tolgono al down
l'hit in L3 sul `mid` appena scritto.  Rimossi.

### Nota sulla deriva della macchina

Nella stessa sessione il prefill e' calato in modo monotono: **56,6 -> 50,6 ->
46,8 -> 40,2 t/s** a parita' di codice (build pulita, `391` corretto).  Non e' una
regressione del codice (gli A/B interni a ogni run restano coerenti) ma uno stato
della macchina che peggiora: ore di run, VM attiva, probabile throttling termico
e reclaim.  **Conseguenza: i confronti fra run distanti nel tempo non valgono
nulla su questa macchina; solo A/B nello stesso run** (e possibilmente con
`uptime`/`fincore` annotati).

### Store del mid con staging L1 (MID_STAGE): +1,2%, dentro il rumore

Idea: 4 byte per (riga, token) in un buffer da 35,6 MB = una linea nuova per
token = un RFO per store; con uno staging da 20 KB in L1 e un flush contiguo per
token le scritture diventano a linea intera.

| | prefill | mid | down |
|---|---|---|---|
| STAGE=0 | 55,22 t/s | 396,3 | 206,2 |
| STAGE=1 | **55,86 t/s** | 392,0 | 203,8 |

**+1,2%, cioe' rumore**, e la spiegazione e' istruttiva: con l'ordine
**chunk-esterno**, per un token fisso le righe avanzano di **1 float per
iterazione**, quindi **16 righe consecutive cadono nella stessa linea di cache** e
il RFO si paga 1 volta su 16 anche senza staging.  Il "costo degli store" che il
bench misurava (-7%) non esiste in questo ordine di loop.

Validato (`391` corretto anche con STAGE=1).  Default **spento**; tenuto solo come
opzione, candidato alla rimozione (non paga il costo di complessita').

### Bilancio delle varianti provate in-modello (riassunto per chi riprende)

| variante | esito in-modello |
|---|---|
| chunk-esterno nel mid | **+5,2%** (tenuta, default) |
| prefetch NTA dei pesi | +10% di tetto nel bench, fasi -2,5%/-4%, t/s nel rumore (tenuta) |
| sinu8 vettoriale | +2% circa (tenuta); **bug di store a 64 byte corretto** |
| 2 righe nel worker | **-12,6% sul mid** (rimossa; il crash era il bug di silu8) |
| store NT nel mid | **-7%** (rimossa) |
| read-ahead MADV_WILLNEED | **-4%** e +35% di disco (rimossa) |
| staging L1 + flush contiguo | +1,2% (rumore, default spento) |
| chunk 16 token | +6% di tetto nel bench, non integrato (rompe la residenza L1) |

### NTA anche sui kernel di ISTA: misurato NEGATIVO, rimosso

Estensione del prefetch NTA (che su IQ2_S/UD vale un mid -2,5%) agli altri quattro
kernel batch (IQ2_XXS, IQ2_XS, IQ3_S, IQ3_XXS), quelli che usa ISTA:

| ISTA | prefill | mid | down |
|---|---|---|---|
| NTA=0 | **49,43 t/s** | 506,4 | 183,2 |
| NTA=1 | 47,62 t/s | **533,4** | 184,1 |

**-3,7% e mid +5,3%**: questi kernel usano griglie a 4 valori in `__m256` e righe
piu' lunghe (~980 B), e il prefetch dell'intera riga non paga (probabilmente
compete con le letture reali invece di anticiparle).  Rimosso; il NTA resta **solo**
nel kernel IQ2_S, dove il guadagno e' misurato.

Nota di metodo: e' il quinto caso in questa sessione in cui una modifica che *sembra*
la stessa cosa su un kernel "gemello" si comporta in modo opposto.  I kernel
batched di questa famiglia **non sono intercambiabili**: vanno misurati uno per uno.

### 2 righe nel kernel del down (IQ4_NL): corretto ma NEUTRO, rimosso

Il down e' la fase con efficienza piu' bassa (76% del tetto) e il traffico per MAC
piu' alto (4 byte/MAC: il vettore `mid` viene riletto per ogni riga di uscita).
Idea: due righe di uscita per passata condividono i load di `mid` (2 byte/MAC).  A
differenza del mid non c'e' assembly per riga da duplicare: era l'unica variante
con meccanismo pulito rimasta.

| UD | prefill | mid | **down** |
|---|---|---|---|
| DOWN_R2=0 | 58,56 t/s | 369,1 | **197,9** |
| DOWN_R2=1 | 56,34 t/s | 393,1 | **198,4** |

**La fase down non cambia** (197,9 vs 198,4 ms): i load condivisi non sono il
collo.  La differenza di t/s (56,3 vs 58,6) e' dovuta al **mid** (369 vs 393 ms),
che questo codice non tocca: e' deriva della macchina (la stessa che in sessione
ha fatto scendere il prefill da 56,6 a 40,2 t/s a parita' di codice).

Correttezza validata (`391` ok con DOWN_R2=1).  Rimosso: era neutro.

**Conclusione**: anche l'ultima idea kernel con meccanismo pulito e' risultata
neutra in-modello.  Delle 7 varianti kernel provate in questa sessione, **1 sola**
(chunk-esterno) ha dato un guadagno reale; tutte le altre sono state neutre o
negative.  E' il motivo per cui la raccomandazione finale e' cambiare livello
(piu' banda/RAM, o meno byte a livello algoritmico), non continuare a cercare
kernel migliori.

### Perche' ISTA e' piu' lenta di UD: il tipo, non il carico (misurato)

`DS4_QWEN4_MOE_COUNT=1` ora stampa anche i tipi e i byte-riga veri:

| | gate/up | down | byte/riga gate | pesi mid/layer | pesi down/layer |
|---|---|---|---|---|---|
| UD | IQ2_S (22) | IQ4_NL (20) | 820 B | 537 MB | 472 MB |
| ISTA | IQ2_XS (17) / IQ2_XXS (16) | IQ4_NL (20) / Q2_0 (42) | 740 / 660 B | 485 / 433 MB | 472 / 236 MB |

Carico **identico** in entrambi (T=1739, ne=512, ns=10, k_in=2560, k_ff=640,
pairs=17390, MACs=56,98 GMAC, act=64961 MB) e ISTA ha **meno** byte di pesi (660-740
B/riga contro 820).  Quindi la spiegazione "piu' byte" **non vale** (era una mia
ipotesi sbagliata, corretta qui).

Rate dei kernel batch, stesso run:

| tipo | L1-hot | streamed |
|---|---|---|
| IQ2_S (UD, VNNI) | 26,76 | **13,00** |
| IQ2_XS (ISTA) | **26,88** | 9,89 |
| IQ2_XXS (ISTA) | 23,74 | 9,37 |
| IQ3_S | 15,11 | 10,26 |

**A caldo i tipi di ISTA sono pari (o migliori)**: il divario esiste **solo nel
regime streamed**, +31-39% per IQ2_S, che coincide con il divario in-modello
(ISTA mid 540 vs UD 390 ms = +38%).  Il modello sta in quel regime: 6,6-8,9
GMAC/s per thread contro 9,4-13,0 dello streamed, cioe' il 68-70% in **entrambi**.

**Conclusione**: il divario UD/ISTA e' **tolleranza alla latenza (MLP) nel regime
streamed**, non byte ne' conteggio istruzioni a caldo.  Il kernel VNNI a 512 bit ha
piu' lavoro indipendente in volo per passo; i kernel `maddubs` a 256 bit di ISTA no.
Premio di una riscrittura in stile VNNI dei kernel IQ2_XXS/IQ2_XS: fino a
**+25-30% sul mid di ISTA** (540 -> ~420 ms), cioe' ISTA da 46 a ~55-58 t/s.
Rischio: il tentativo VNNI su questi tipi fu gia' misurato negativo una volta, ma
allora il bersaglio era l'assembly, mentre la misura di oggi dice che il collo e'
il **ciclo dei token** (MLP), non l'assembly.

## BUG DI CORRETTEZZA nel kernel batch IQ2_XXS (trovato e corretto)

**Sintomo**: `IQ2_XXS batch x8 vs singolo: worst rel diff 1.11e+01` (1110% di errore).

**Causa**: nel kernel `qwen4_cpu_dot_iq2_xxs_q8k_batch` mancava l'offset `ib32` sul
puntatore delle attivazioni:
```c
const int8_t *q8 = ys[b][i].qs;              /* SBAGLIATO */
const int8_t *q8 = ys[b][i].qs + 32*ib32;    /* corretto (come IQ3_S) */
```
Ogni iterazione del ciclo consuma 16 byte di pesi = 64 valori, quindi le attivazioni
vanno lette da `ib32*32` (32 valori per sottoblocco).  Senza offset il kernel
rileggeva i **primi 64 valori per tutte e 4 le iterazioni** del blocco: 3/4 delle
attivazioni del blocco venivano ignorate e sostituite dalle prime.

**Impatto**: IQ2_XXS e' il tipo di **9 dei 48 layer MoE** del modello ISTA-GSQ-RCO
(quello in uso).  Quei layer producevano contributi MoE sbagliati in ogni prefill e
in ogni token di decode.  Il difetto **non era coperto da nessun test**: `check_q8k`
valida il percorso a **token singolo**, e il percorso **batch** era verificato solo
per IQ2_S.

**Perche' non si vedeva**: il MoE somma 10 esperti su 512 per token, quindi un
contributo sbagliato su 9 layer viene diluito; `391` continuava a passare.

**Correzione e verifica**:
- offset aggiunto; tutti e cinque i kernel batch ora danno `0.00e+00` contro il
  kernel a token singolo (bit-identici);
- `391` corretto su **ISTA** (con i 9 layer ora giusti) e su **UD** (controllo);
- test `test_qwen4_cpu_dot`: **ok**.

**Test aggiunto (da eseguire sempre)**: `check_batch_any()` in
`tests/test_qwen4_cpu_dot.c` confronta, per ognuno dei 5 tipi, il kernel **batch**
(percorso di produzione del prefill) col kernel a **token singolo** (che e' validato
contro il riferimento indipendente).  Prima di questo controllo il percorso batch
era scoperto per 4 tipi su 5.

**Lezione**: quando un kernel di produzione non e' coperto da un test che lo
confronta con una implementazione indipendente, un errore puo' restare in produzione
per settimane indisturbato.  Il primo sospetto e' nato dal leggere il codice per
*ottimizzarlo*, non da un fallimento: **leggere il percorso di produzione come se
fosse nuovo e' un'attivita' di qualita', non solo di performance.**

### Copertura chiusa anche per il down: 7 kernel batch verificati, nessun altro bug

Dopo il bug di IQ2_XXS ho verificato l'altro percorso di produzione senza copertura:
i kernel **fp32 batch del down** (IQ4_NL, Q2_0), usati da **tutti i 48 layer** del
modello (Q2_0 in 30, IQ4_NL in 18).  Comparivano solo in `time_down_mt` (timing),
mai confrontati col kernel a token singolo.

Aggiunto `check_fp32_batch()`: confronta il kernel batch fp32 col kernel a token
singolo, a k=640 e k=2560.  Risultato:

```
IQ4_NL   fp32 batch x8 vs singolo: 0.00e+00 ok   (k=640 e k=2560)
Q2_0     fp32 batch x8 vs singolo: 0.00e+00 ok   (k=640 e k=2560)
```

**Stato della copertura dei kernel di produzione** (tutti a `0.00e+00` contro il
percorso a token singolo, che e' validato contro un riferimento indipendente in
doppia precisione):

| kernel batch | tipo | layer ISTA | esito |
|---|---|---|---|
| `iq3_s_q8k_batch` | IQ3_S | 13 | ok |
| `iq2_xs_q8k_batch` | IQ2_XS | 10 | ok |
| `iq2_xxs_q8k_batch` | IQ2_XXS | 9 | ok (era **rotto**, corretto) |
| `iq2_s_q8k_vnni_batch` | IQ2_S | 10 | ok |
| `iq3_xxs_q8k_batch` | IQ3_XXS | 6 | ok |
| `iq4_nl_batch` (fp32) | IQ4_NL | down di 18 layer | ok |
| `q2_0_batch` (fp32) | Q2_0 | down di 30 layer | ok |

Da eseguire **sempre** dopo ogni modifica a questi kernel:
```sh
make test-qwen4-cpu-dot && ./tests/test_qwen4_cpu_dot | grep -E "vs singolo|fp32 batch"
```

### Mappa completa dei tipi del modello ISTA (dopo il fix)

Layer MoE per tipo e rate dei kernel batch (L1-hot = stabile, streamed = oscilla
molto fra run: IQ2_S ha dato 13,00 e poi 6,94 nello stesso giorno, quindi **solo i
numeri hot sono confrontabili**):

| tipo | kernel | layer | L1-hot | streamed (indicativo) |
|---|---|---|---|---|
| IQ2_S | `iq2_s_q8k_vnni_batch` (**VNNI**) | 10 | 26,8-30,6 | 6,9-13,0 |
| IQ3_S | `iq3_s_q8k_batch` | **13** | **14,9-15,1** | 8,8-10,3 |
| IQ2_XXS | `iq2_xxs_q8k_batch` | 9 | 23,0-23,7 | 8,9-9,4 |
| IQ2_XS | `iq2_xs_q8k_batch` | 10 | 21,9-26,9 | 8,8-9,9 |
| IQ3_XXS | `iq3_xxs_q8k_batch` | 6 | 21,4 | 10,0 |
| IQ4_NL | `iq4_nl_batch` (fp32, down) | down di 18 | (misurato nel MT: 34/GM) | — |
| Q2_0 | `q2_0_batch` (fp32, down) | down di 30 | — | — |

**Diagnosi per IQ3_S** (il piu' usato e l'unico fuori scala a caldo):
512 MAC per passo costano ~146 cicli (= 3,5 MAC/ciclo, ~0,8 istruzioni/ciclo su una
macchina che ne fa 3-4): non e' limite di throughput ma di **catena di dipendenze
per passo** - build degli indici (`sllv`+mask+or) -> 16 load scalari di griglia ->
`set_epi32` -> `maddubs` -> `madd` -> `add`.  Ogni passo dipende dal precedente e
non c'e' lavoro indipendente che lo copra.

Il rimedio e' far sovrapporre due passi (unroll x2), ma con 8 token gli accumulatori
occupano gia' **16 registri ymm** (le coppie `sumi1`/`sumi2`), quindi il corpo
duplicato va in spill.  Una riscrittura va quindi progettata insieme a uno dei due:
- accumulo in un **singolo** vettore int32 per token (8 registri invece di 16) e
  `dpwssd` per i prodotti, liberando registri per l'unroll;
- oppure 4 token per chiamata con due passi sovrapposti.
Entrambe vanno misurate nel bench con `DS4_BENCH_DOT=1` e validate con
`check_batch_any`/`391` prima di ogni integrazione.
