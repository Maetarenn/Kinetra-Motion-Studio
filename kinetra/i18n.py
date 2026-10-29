"""Tiny runtime i18n: JSON locale files in kinetra/locales, English is the master and the fallback."""
import json
import locale
from pathlib import Path

LOCALE_DIR = Path(__file__).resolve().parent / "locales"
_tables, _lang = {}, "en"


def _load(code):
    if code not in _tables:
        f = LOCALE_DIR / f"{code}.json"
        _tables[code] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return _tables[code]


def available():
    """{code: native language name}, English first."""
    out = {p.stem: _load(p.stem).get("_name", p.stem) for p in sorted(LOCALE_DIR.glob("*.json"))}
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "en", kv[1])))


def set_language(code):
    global _lang
    _lang = code if (LOCALE_DIR / f"{code}.json").exists() else "en"
    return _lang


def get_language():
    return _lang


def system_language():
    try:
        code = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "en")[:2].lower()
    except Exception:
        code = "en"
    return code if (LOCALE_DIR / f"{code}.json").exists() else "en"


def t(key, *args):
    s = _load(_lang).get(key) or _load("en").get(key) or key
    if args:
        try:
            return s.format(*args)
        except (IndexError, KeyError):
            return s
    return s
