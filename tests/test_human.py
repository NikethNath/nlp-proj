"""Collection kit: form titles, CSV ingestion, flags and divergence analysis.

Uses a toy manifest and a hand-written export shaped like Google Forms output,
so no FLORES data or Google account is needed.
"""
import csv
import json
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from latscript.human import analyze, forms, ingest

MANIFEST = {
    "split": "devtest",
    "anchors": [1, 2],
    "blocks": [[3], [4]],
    "sentences": {
        "ml": {"1": "നമസ്കാരം കേരളം", "2": "എന്റെ വീട്", "3": "പുസ്തകം വായിക്കുക", "4": "ഞങ്ങളുടെ കുട്ടി"},
        "ta": {"1": "வணக்கம் சென்னை", "2": "என் வீடு", "3": "புத்தகம் படி", "4": "எங்கள் குழந்தை"},
    },
}
ENGLISH = {1: "Hello Kerala", 2: "My house", 3: "Read the book", 4: "Our child"}


def title(idx):
    return forms.sentence_title("ml", idx, MANIFEST["sentences"]["ml"][str(idx)])


def write_export(path, rows):
    background = [t.format(lang="Malayalam") for t, _ in forms.BACKGROUND.values()]
    header = (["Timestamp", forms.CONSENT_TITLE, forms.CODE_TITLE] + background
              + [title(1), title(2), forms.SET_TITLE, title(3), title(4)])
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def row(code, a1, a2, set_no, b3="", b4="", consent=forms.CONSENT_CHOICE):
    return {"Timestamp": "2026/09/14 10:00:00", forms.CONSENT_TITLE: consent,
            forms.CODE_TITLE: code, title(1): a1, title(2): a2,
            forms.SET_TITLE: set_no, title(3): b3, title(4): b4}


def test_rendered_titles_match_ingest_header_pattern():
    script = forms.render(MANIFEST)
    titles = re.findall(r'"(\[(?:ml|ta)-\d{4}\][^"]*)"', script)
    assert len(titles) == 8
    assert all(ingest._HEADER.match(t) for t in titles)


def test_ingest_maps_answers_and_flags(tmp_path):
    path = tmp_path / "ml.csv"
    write_export(path, [
        row(" ml01", "namaskaram keralam", "ente veedu", "1", b3="pusthakam vaayikkuka"),
        row("ML02", "namaskaaram keralam", "ente veed", "2", b4="njangalude kutti"),
        row("ML03", "നമസ്കാരം കേരളം", "my house", "1", b3="book"),
        row("ML02", "namaskaram kerlam", "ente vid", "2", b4="ngalude kuti"),
    ])
    out = ingest.ingest(path, "ml", MANIFEST, english=ENGLISH)
    by = {(r.respondent, r.submission, r.idx): r for r in out}

    first = by[("ML01", 1, 3)]
    assert (first.role, first.block, first.typed) == ("block", 1, "pusthakam vaayikkuka")
    assert by[("ML01", 1, 1)].role == "anchor" and by[("ML01", 1, 1)].flags == []

    assert "native_script" in by[("ML03", 3, 1)].flags
    assert "translated" in by[("ML03", 3, 2)].flags
    assert "length" in by[("ML03", 3, 3)].flags
    assert all("repeat_code" in r.flags for r in out if r.respondent == "ML02")
    assert len(out) == 12


def test_analyze_word_disagreement_and_exclusions(tmp_path):
    path = tmp_path / "ml.csv"
    write_export(path, [
        row("ML01", "namaskaram keralam", "ente veedu", "1", b3="pusthakam vaayikkuka"),
        row("ML02", "namaskaaram keralam", "ente veedu", "2", b4="njangalude kutti"),
        row("ML03", "നമസ്കാരം കേരളം", "my house", "1", b3="book"),
    ])
    responses = ingest.ingest(path, "ml", MANIFEST, english=ENGLISH)
    flores_rows = {
        1: {"l3_writers": ["namaskaram keralam", "namaskaram keralam"],
            "l2_canonical": "namaskaram keralam", "l2_lexical": "namaskaaram keralam"},
        2: {"l3_writers": ["ente veedu", "ente veed"],
            "l2_canonical": "ente veedu", "l2_lexical": "ente veedu"},
    }
    report = analyze.analyze(responses, flores_rows)

    # All three of ML03's answers are excluded (native script, translated, too
    # short), leaving two writers per anchor: 1 of 2 words differ on sentence 1,
    # 0 of 2 on sentence 2.
    assert report["sentences_with_2plus_writers"] == 2
    assert report["human"]["word_disagreement"] == 0.25
    assert report["simulated_l3"]["word_disagreement"] == 0.25
    assert report["per_respondent"]["ML03"]["excluded"] == 3
    json.dumps(report, ensure_ascii=False)
