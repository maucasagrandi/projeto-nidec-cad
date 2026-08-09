import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "validation" / "gdt" / "reference_catalog.json"
SOURCE_ROOT = PROJECT_ROOT / "cotas"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _normalize_filename(name: str) -> str:
    return name.strip().casefold().replace(" .", ".")


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _active_entries() -> list[dict]:
    return [
        entry
        for entry in _catalog().get("entries", [])
        if str(entry.get("status", "active")).lower() == "active"
    ]


def test_every_reference_image_is_registered():
    registered = {
        _normalize_filename(str(entry["source"]))
        for entry in _active_entries()
    }
    source_files = {
        _normalize_filename(path.name)
        for path in SOURCE_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    }

    assert source_files == registered


def test_every_registered_source_exists():
    available = {
        _normalize_filename(path.name)
        for path in SOURCE_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    }

    for entry in _active_entries():
        assert _normalize_filename(str(entry["source"])) in available


def test_target_paths_are_unique():
    targets = [
        (str(entry["class_name"]).strip().lower(), str(entry["target_name"]))
        for entry in _active_entries()
    ]
    assert len(targets) == len(set(targets))


def test_negative_controls_are_not_an_active_characteristic():
    classes = {
        str(entry["class_name"]).strip().lower()
        for entry in _active_entries()
    }
    assert "negative_controls" not in classes


def test_concentricity_and_coaxiality_share_visual_class():
    by_source = {
        _normalize_filename(str(entry["source"])): str(entry["class_name"]).strip().lower()
        for entry in _active_entries()
    }

    assert by_source["concentricity.png"] == "concentricity_coaxiality"
    assert by_source["coaxiality.png"] == "concentricity_coaxiality"


def test_roundness_uses_canonical_circularity_class():
    by_source = {
        _normalize_filename(str(entry["source"])): str(entry["class_name"]).strip().lower()
        for entry in _active_entries()
    }

    assert by_source["roundness.png"] == "circularity"
