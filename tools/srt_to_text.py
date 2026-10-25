"""Convert Panopto/Zoom SRT caption files to plain paragraphs.

Drops cue numbers, timestamps and the auto-generated notice, merges consecutive
cues from the same speaker, and relabels speakers: the instructor as "Instructor",
everyone else as "Student".

Usage:
    python srt_to_text.py out_dir in1.txt [in2.txt ...]
Several inputs given with --concat are joined into one output named after the first.
"""
import re
import sys
from pathlib import Path

INSTRUCTOR = re.compile(r"^(?:Audio shared by )?David A Knowles:\s*")
SPEAKER = re.compile(r"^([A-Z][A-Za-z.' -]{1,40}):\s+")
TIME = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> ")


def cues(path):
    for block in Path(path).read_text(encoding="utf-8-sig").split("\n\n"):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        lines = [l for l in lines if not l.strip().isdigit() and not TIME.match(l)]
        text = " ".join(lines).strip()
        text = text.replace("[Auto-generated transcript. Edits may have been applied for clarity.]", "").strip()
        if text:
            yield text


def paragraphs(paths):
    out, cur_spk, cur = [], None, []
    for p in paths:
        for t in cues(p):
            if INSTRUCTOR.match(t):
                spk, t = "Instructor", INSTRUCTOR.sub("", t)
            elif SPEAKER.match(t):
                spk, t = "Student", SPEAKER.sub("", t)
            else:
                spk = cur_spk or "Speaker"
            if spk != cur_spk and cur:
                out.append(f"{cur_spk}: " + " ".join(cur))
                cur = []
            cur_spk = spk
            cur.append(t)
    if cur:
        out.append(f"{cur_spk}: " + " ".join(cur))
    return "\n\n".join(out) + "\n"


if __name__ == "__main__":
    args = sys.argv[1:]
    concat = "--concat" in args
    args = [a for a in args if a != "--concat"]
    out_dir, inputs = Path(args[0]), args[1:]
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [inputs] if concat else [[i] for i in inputs]
    for g in groups:
        name = re.sub(r"_\d{4}$", "", Path(g[0]).stem) + ".txt"
        (out_dir / name).write_text(paragraphs(g), encoding="utf-8")
        print(out_dir / name)
