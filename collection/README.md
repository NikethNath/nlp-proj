# Collecting human-typed romanizations

Native speakers see Malayalam or Tamil FLORES sentences and type each one in
English letters, the way they would on WhatsApp. The result is naturalistic
romanization aligned sentence-by-sentence with L1, L2 and the simulated L3, and
a direct measurement of how much real writers disagree on spelling.

| File | What it is |
|---|---|
| `manifest.json` | The 248 sentences per language and how they are grouped. The same FLORES sentences are used for both languages |
| `create_forms.gs` | Google Apps Script that builds both Google Forms |

Both are generated; rebuild with `python -m latscript.human.select` and
`python -m latscript.human.forms` (with `PYTHONPATH=src`). Changing the
manifest after collection starts invalidates earlier responses.

## Design

Each respondent types **8 anchor sentences**, the same for everyone, and
**one set of 12**, their own: 20 sentences, about 20-25 minutes on a phone.
There are 20 sets per language.

- Anchors give many writers per sentence, which is the inter-writer divergence
  measurement.
- Sets give coverage. Hand them out in order; after set 20, start again at set
  1, which adds a second writer to those sentences.

| Respondents per language | Anchor writers | Set sentences covered |
|---|---|---|
| 5 | 5 each | 60 |
| 10 | 10 each | 120 |
| 15 | 15 each | 180 |
| 20 | 20 each | 240 (all) |
| 30 | 30 each | 240, of which 120 with two writers |

Divergence statistics need at least 5 respondents per language, and 10 is
comfortable.

## Before you start

- **Ethics.** If this work is for a course or a university, check whether it
  needs ethics approval before you collect anything.
- **Privacy.** The form collects no names, emails or phone numbers.
  Respondents are identified only by a code. The only record of which code
  belongs to whom is your tracker. Keep it private and never commit it.
- **Pilot.** Run the whole procedure with two people first. Time them, then
  ingest their CSV to check the pipeline end to end.

## Steps

1. **Create the forms.** Go to https://script.google.com, create a new project,
   replace its contents with `create_forms.gs`, and run `createForms`. Approve
   the permissions prompt. The execution log shows, per language, the link to
   send, the edit link and the response sheet.
2. **Make a private tracker** with columns `code`, `language`, `set`, `sent on`
   and `done`. Codes are `ML01`, `ML02`, … and `TA01`, … Assign sets 1-20 in
   order, per language.
3. **Send each person their link, code and set number.** For example:

   > Hi! I'm collecting data for a research project on how people type
   > Malayalam in English letters. It takes about 20 minutes on your phone.
   > There are no right or wrong spellings, just type the way you normally do.
   > Your code is **ML07** and your set is **7**. Link: …

4. **Export responses.** In each response sheet choose File → Download →
   Comma-separated values, and save as `data/raw/human/ml.csv` or
   `data/raw/human/ta.csv`. That directory is not committed.
5. **Ingest and analyze:**

   ```bash
   PYTHONPATH=src .venv/bin/python -m latscript.human.ingest ml data/raw/human/ml.csv
   PYTHONPATH=src .venv/bin/python -m latscript.human.analyze ml
   ```

   Results go to `data/derived/human/`.

## Flags raised on ingest

Nothing is dropped at ingest. Analysis excludes the first four flags.

| Flag | Meaning |
|---|---|
| `native_script` | Typed mostly in Malayalam or Tamil letters (pasted, or a transliteration keyboard was on) |
| `length` | Much shorter or longer than any romanization of the sentence |
| `translated` | Mostly words from the English translation: translated rather than typed |
| `no_consent` | The consent box is not ticked |
| `repeat_code` | The same code was submitted more than once. Check your tracker |
| `set_mismatch` | The answers belong to a different set than the one chosen |
