# Hand check of the generated corpus

`sanity_check_ml.csv` and `sanity_check_ta.csv` hold 25 randomly chosen FLORES
devtest sentences each, rendered at every orthographic level. Open them in
Google Sheets or Excel, fill the four empty columns, and save back as CSV.
Checking one language takes about 45 minutes.

| Column | What to enter |
|---|---|
| `l2_canonical_ok` | `Y` if the IndicXlit romanization is a spelling a careful typist could produce for every word, else `N` |
| `l2_lexical_ok` | Same question for the Dakshina-based romanization |
| `l3_bad_words` | Words in any of the five writer columns that no real person would type, separated by `;`. Leave empty if all are plausible |
| `notes` | Anything systematic: a letter pattern that is always wrong, a word that means something else, and so on |

Judge plausibility, not correctness. `ippo`, `ippol` and `ipol` are all fine;
something like `vnkkm` is not. Unusual spellings of foreign names are expected.

Your native language is the one to judge. For the other, mark only what is
clearly broken, and say so in `notes`.
