"""Regression guard for code-review finding #1: the repo must not re-acquire
'official MLPerf' overclaims, and must keep the honesty disclaimers."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Phrases that would misrepresent these smoke tests as official/conformant MLPerf.
# We only forbid them in AFFIRMATIVE use; a line that negates ("NOT the official MLPerf way")
# is allowed, so we match the phrase and require the line to not also contain a negator.
FORBIDDEN = [
    'the "official MLPerf way"',
    "the official MLPerf way",
    "*real MLPerf*",
    "a *real MLPerf*",
]
NEGATORS = ("not ", "n't", "unofficial", "inspired", "≠", "!=", "NOT ")


def _md_files():
    return [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]


def test_no_affirmative_official_mlperf_claims():
    offenders = []
    for p in _md_files():
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            low = line.lower()
            for phrase in FORBIDDEN:
                if phrase.lower() in low and not any(n.lower() in low for n in NEGATORS):
                    offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "Affirmative 'official/real MLPerf' claims re-appeared:\n" + "\n".join(offenders)


def test_readme_has_not_official_disclaimer():
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore")
    assert "NOT official MLPerf" in readme or "not official MLPerf" in readme.lower()
    assert "MLPerf-*inspired*" in readme or "MLPerf-inspired" in readme


def test_whisper_documented_as_not_loadgen():
    """Whisper genuinely uses no LoadGen — the docs must say so (finding #1 core)."""
    arch = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8", errors="ignore").lower()
    assert "no loadgen" in arch or "not even loadgen" in arch or "custom loop" in arch
