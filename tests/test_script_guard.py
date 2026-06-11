import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LR-Agent-analysis"))

import pytest
from script_guard import ScriptGuardError, sanitize_script_text, validate_script


def test_validate_allows_stdlib_script():
    validate_script(
        "import json\nfrom pathlib import Path\n"
        "data = json.loads(Path('annotations.json').read_text(encoding='utf-8'))\n"
        "print(len(data.get('files', [])))"
    )


def test_validate_rejects_subprocess():
    with pytest.raises(ScriptGuardError):
        validate_script("import subprocess\nsubprocess.run(['ls'])")


def test_sanitize_removes_lone_surrogates():
    dirty = "print('ok')\n" + "\udcad"
    cleaned = sanitize_script_text(dirty)
    assert "\udcad" not in cleaned
    validate_script(cleaned)


def test_validate_allows_data_global_variable_script():
    validate_script(
        "label_counts = DATA['labelCounts']\n"
        "for label, count in sorted(label_counts.items()):\n"
        "    print(f'{label}: {count}')"
    )


def test_validate_rejects_open_call():
    with pytest.raises(ScriptGuardError):
        validate_script("with open('annotations.json') as f:\n    print(f.read())")


def test_validate_allows_collections_counter():
    validate_script(
        "from collections import Counter\n"
        "c = Counter(DATA['labelCounts'])\n"
        "print(dict(c.most_common(3)))"
    )
