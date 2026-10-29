import json, re
from pathlib import Path
from kinetra import i18n

PH = re.compile(r"\{\d+(?::[^}]*)?\}")


def test_all_locales_complete_and_placeholders_match():
    en = json.loads((i18n.LOCALE_DIR / "en.json").read_text(encoding="utf-8"))
    codes = list(i18n.available())
    assert len(codes) >= 8
    for code in codes:
        d = json.loads((i18n.LOCALE_DIR / f"{code}.json").read_text(encoding="utf-8"))
        assert set(d) == set(en), (code, set(en) ^ set(d))
        for k, v in en.items():
            if k != "_name":
                assert sorted(PH.findall(d[k])) == sorted(PH.findall(v)), (code, k)


def test_fallback_and_format():
    i18n.set_language("tr")
    assert i18n.t("btn.start") == "BAŞLAT"
    assert i18n.t("log.cam_online", 1, "x") == "Kamera 1 açık: x"
    assert i18n.t("does.not.exist") == "does.not.exist"
    i18n.set_language("xx")
    assert i18n.get_language() == "en"


def test_every_ui_key_exists():
    en = json.loads((i18n.LOCALE_DIR / "en.json").read_text(encoding="utf-8"))
    used = set()
    for f in Path(i18n.__file__).parent.rglob("*.py"):
        used |= set(re.findall(r"""\bt\(\s*["']([a-z_]+\.[a-z_]+)["']""", f.read_text(encoding="utf-8")))
        used |= set(re.findall(r"""_msg\(\s*["']([a-z_]+\.[a-z_]+)["']""", f.read_text(encoding="utf-8")))
    missing = {k for k in used if k not in en and k not in ("log.face_event",)}   # face events resolve to fe.* at runtime
    assert not missing, missing
