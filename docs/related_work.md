# Related work

Survey done before running Phase 2, to check that the script-versus-spelling
decomposition has not already been published. The closest papers are listed
first. Last checked October 2026.

## The two strands this project sits between

**Strand A: romanization helps LLMs.** These papers use *canonical*, machine-generated romanization.

- **RomanSetu** (Husain et al., ACL 2024) continually pretrains Llama 2 on IndicXlit-romanized text of Indic languages. It finds that romanized inputs:
  - align better with English representations,
  - roughly halve sequence length (native-script fertility is at least 2x that of romanized text),
  - improve NLU, NLG and MT.

  Their romanization is a single deterministic form per word: our L2.
- **RomanLens** (Saji et al., 2025) finds that LLMs internally pass through a latent romanized representation when processing non-Latin scripts.

**Strand B: romanized user text hurts LLMs.** These papers use *natural or noisy* romanization.

- **Script Gap** (Khullar et al., 2025, arXiv:2512.10780) uses 3,153 real WhatsApp maternal-health messages in Hindi, Telugu, Kannada, Marathi, Punjabi and Nepali, with 8 LLMs.
  - Romanized messages score 1–8 weighted-F1 points below native script for top models, and up to 24 for Nepali.
  - A parallel HealthBench variant, romanized by GPT-4o, shows a mean 5.4-point drop.
  - Converting romanized messages back to native script recovers GPT-4o from 75.9 to 80.1 weighted F1.
  - The paper also reports a flip rate, but across *sampling temperatures*, not across spellings.
  - It does not separate the script effect from spelling variation. Its own words for the residual are "orthographic noise".
  - It includes no Malayalam or Tamil.
- **Indi-RomCoM** (Das et al., 2026, arXiv:2606.30790) is an instruction benchmark in Hindi, Bengali, Gujarati and Tamil, with three code-mixing levels and 17–19 models.
  - Romanization comes from Aksharantar / Aksharamukha. The authors keep "natural transliteration variability" but do not vary it as an experimental factor.
  - The manipulated variable is the *code-mixing density*, not spelling.
  - Few-shot prompting gives +7–8 points; LoRA on code-mixed data helps.
- **Script Sensitivity** (2026, arXiv:2601.14958) measures perplexity of 24 open LMs on Sinhala in Unicode, romanized and mixed scripts. Median perplexity is more than 300x worse on romanized text.
- **Beyond Specialization** (2025, arXiv:2505.19851) benchmarks LLMs as *transliterators*. Synthetic spelling noise added to romanized words hurts IndicXlit more than a fine-tuned GPT-4o.

**The tension.** Strand A says that putting Indic text in Latin script is good for LLMs. Strand B says that real Latin-script Indic text is bad for LLMs. Both can be true if the *script* is not the problem and the *spelling variance* is. No paper we found:
1. holds content fixed and measures native vs. canonical romanization vs. naturally varied romanization on the same sentences, or
2. measures how stable a model's output is across different spellings of the same sentence.

These are the two contributions this project tests: the L1/L2/L3 decomposition and the orthographic consistency metric.

## Data and tools we build on

- **Dakshina** (Roark et al., LREC 2020): native words with several human romanizations each. It reports that the modal spelling matches another annotator only 69% (ml) / 63% (ta) of the time; we reproduce 69.5% / 64.6%.
- **Context-aware transliteration of romanized South Asian languages** (Kirov et al., Computational Linguistics 2024) documents the extent of spelling variation in Dakshina and its ambiguity (Tamil *puli* means both tamarind and tiger).
- **Aksharantar / IndicXlit** (Madhani et al., Findings of EMNLP 2023): transliteration pairs and the model we use for L2 and back-transliteration.
- **DravidianCodeMix** (Chakravarthi et al., 2021, 2022): romanized, code-mixed YouTube comments with offensive-language labels. The 2021 shared-task overview already proposed transliterating comments to native script before classification, which is our first mitigation rung.
- **FLORES-200** (NLLB Team, 2022): parallel sentences with professional English references.
- **MuRIL** (Khanuja et al., 2021): an Indic encoder pretrained with transliterated text.
- **Romanized-to-native Malayalam transliteration** (IndoNLP 2025): a BiLSTM back-transliterator trained on Dakshina + Aksharantar, an alternative to IndicXlit for the back-transliteration rung.

## Verdict

The headline observation, that LLMs are worse on romanized Indic text, is established (Strand B). Romanization as a fixed transform being helpful is also established (Strand A). Still open, as far as this survey can tell:
- the decomposition into script and spelling terms on parallel data,
- a consistency metric across spellings of the same input,
- a cost-annotated mitigation ladder,
- any of this for Malayalam, which neither close paper covers.

## Sources

- https://arxiv.org/abs/2401.14280 (RomanSetu)
- https://arxiv.org/abs/2502.07424 (RomanLens)
- https://arxiv.org/abs/2512.10780 (Script Gap)
- https://arxiv.org/abs/2606.30790 (Indi-RomCoM)
- https://arxiv.org/abs/2601.14958 (Script Sensitivity, Sinhala)
- https://arxiv.org/abs/2505.19851 (Beyond Specialization)
- https://direct.mit.edu/coli/article/50/2/475/119145 (Context-aware transliteration)
- https://aclanthology.org/2025.indonlp-1.20/ (Romanized-to-native Malayalam)
- https://aclanthology.org/2021.dravidianlangtech-1.3.pdf (Offensive language identification for Dravidian languages)
