import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.demo import demo_embed, demo_generate, mark_sample_payload, resolve_demo_mode
from app.rag.config import Settings
from app.rag.prompt import build_messages
import main


def test_demo_embed_is_deterministic_and_normalized():
    a = demo_embed(["query: su kayıp oranı"])
    b = demo_embed(["query: su kayıp oranı"])
    assert a.shape == (1, 384)
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_demo_generate_reads_evidence_json():
    evidence = [{"filename": "rapor.md", "page": None, "text": "Su kayıp oranı yüzde 22'ye düştü."}]
    messages = build_messages("Su kayıp oranı nedir?", evidence)
    answer = demo_generate(messages, model_name="demo", max_new_tokens=32)
    assert "22" in answer
    assert "[Kanıt 1]" in answer


def test_resolve_demo_mode_respects_explicit_flags():
    assert resolve_demo_mode(Settings(demo_mode=True)) is True
    assert resolve_demo_mode(Settings(demo_mode=False)) is False


def test_mark_sample_payload_labels_answers():
    marked = mark_sample_payload({"answer": "22 [Kanıt 1]", "sources": []})
    assert marked["sample_data"] is True
    assert marked["answer"].startswith("[sample data]")


def test_demo_app_health_and_ask_use_sample_data(monkeypatch, tmp_path):
    monkeypatch.setenv("EVIDENCERAG_DEMO", "true")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # Rebuild pipeline under demo settings without re-importing the whole module graph.
    settings = Settings(demo_mode=True, index_dir=str(tmp_path / "index"), relevance_threshold=0.8)
    assert resolve_demo_mode(settings) is True
    pipeline = main._build_pipeline(settings, True)
    monkeypatch.setattr(main, "rag_pipeline", pipeline)
    monkeypatch.setattr(main, "_demo_active", True)
    with TestClient(main.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        main.app.state.demo_mode = True
        health = client.get("/health").json()
        assert health["demo_mode"] is True
        assert health["sample_data"] is True
        # Seed happens in lifespan; if empty, ask still returns labeled insufficient/sample.
        answer = client.post("/ask", json={"question": "Yeşil Vadi su kayıp oranı yüzde kaç?"})
        assert answer.status_code == 200
        body = answer.json()
        assert body["sample_data"] is True
        assert "sample data" in body["answer"].casefold()
