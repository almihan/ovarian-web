"""Production defaults must fail closed and scheduling must remain opt-in."""

from backend.config import get_settings


def clear_modes(monkeypatch):
    for name in ("APP_ENV", "RAILWAY_ENVIRONMENT_ID", "RAILWAY_SERVICE_ID",
                 "RAILWAY_PUBLIC_DOMAIN", "PUBLIC_PRECOMPUTED_ONLY", "MONTHLY_UPDATES_ENABLED"):
        monkeypatch.delenv(name, raising=False)


def test_railway_detection_defaults_to_saved_results(monkeypatch):
    clear_modes(monkeypatch)
    monkeypatch.setenv("RAILWAY_SERVICE_ID", "test-service")
    selected = get_settings()
    assert selected.environment == "production"
    assert selected.public_precomputed_only is True
    assert selected.monthly_updates_enabled is False


def test_local_mode_keeps_arbitrary_pmid_processing(monkeypatch):
    clear_modes(monkeypatch)
    selected = get_settings()
    assert selected.environment == "development"
    assert selected.public_precomputed_only is False
    assert selected.monthly_updates_enabled is False


def test_explicit_environment_switches(monkeypatch):
    clear_modes(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    assert get_settings().public_precomputed_only is True
    monkeypatch.setenv("PUBLIC_PRECOMPUTED_ONLY", "false")
    monkeypatch.setenv("MONTHLY_UPDATES_ENABLED", "true")
    selected = get_settings()
    assert selected.public_precomputed_only is False
    assert selected.monthly_updates_enabled is True


def test_modal_trigger_is_cpu_only_and_disabled_by_default(monkeypatch):
    import requests
    import modal_app

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled schedule attempted an HTTP request")

    monkeypatch.delenv("MODAL_MONTHLY_SCHEDULE_ENABLED", raising=False)
    monkeypatch.setattr(requests, "post", forbidden)
    assert modal_app.monthly_update_trigger.local() == {"status": "disabled"}


def test_enabled_modal_trigger_uses_private_header(monkeypatch):
    import requests
    import modal_app

    captured = {}

    class Accepted:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "accepted"}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Accepted()

    monkeypatch.setenv("MODAL_MONTHLY_SCHEDULE_ENABLED", "true")
    monkeypatch.setenv("CORPUS_UPDATE_BASE_URL", "https://example.up.railway.app/")
    monkeypatch.setenv("MONTHLY_UPDATE_TOKEN", "offline-test-token")
    monkeypatch.setattr(requests, "post", fake_post)
    assert modal_app.monthly_update_trigger.local() == {"status": "accepted"}
    assert captured["url"] == "https://example.up.railway.app/api/internal/corpus-updates"
    assert captured["headers"] == {"X-Corpus-Update-Token": "offline-test-token"}
    assert captured["json"] == {"dry_run": False}


def test_reference_seed_preserves_existing_volume_files(tmp_path):
    from backend.services.reference_resources import seed_reference_resources

    seeds = tmp_path / "seeds"
    active = tmp_path / "active"
    seeds.mkdir()
    active.mkdir()
    (seeds / "hgnc_complete_set.txt").write_text("seed", encoding="utf-8")
    (seeds / "chebi_compounds.tsv.gz").write_bytes(b"seed-data")
    (active / "hgnc_complete_set.txt").write_text("existing", encoding="utf-8")
    seed_reference_resources(seeds, active)
    assert (active / "hgnc_complete_set.txt").read_text() == "existing"
    assert (active / "chebi_compounds.tsv.gz").read_bytes() == b"seed-data"
    assert not list(active.glob("*.tmp"))
