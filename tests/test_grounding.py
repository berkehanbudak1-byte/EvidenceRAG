import json
from pathlib import Path

import pytest

from app.rag.grounding import verify_answer


EVIDENCE = [
    {"text": "Su kayıp oranı yüzde 31'den yüzde 22'ye düştü. Proje 86 milyon liraya mal oldu."},
    {"text": "Tramvay günde 21.500 yolcu taşıyor. Kişi başı yeşil alan 5,8 metrekare."},
]


def test_supported_answer_has_no_findings():
    result = verify_answer(
        "Su kaybı %31'den %22'ye düştü [Kanıt 1]; tramvay günde 21500 yolcu taşıyor [Kanıt 2].",
        EVIDENCE,
    )

    assert result == {
        "citations": [1, 2], "invalid_citations": [],
        "unsupported_numbers": [], "question_only_numbers": [],
    }


def test_reports_numbers_missing_from_evidence():
    result = verify_answer("Proje 90 milyon liraya mal oldu ve 86 milyonluk kısmı... [Kanıt 1]", EVIDENCE)

    assert result["unsupported_numbers"] == ["90"]


def test_reports_citations_to_missing_evidence():
    result = verify_answer("Yeşil alan 5,8 metrekare [Kanıt 2] [Kanıt 4].", EVIDENCE)

    assert result["citations"] == [2]
    assert result["invalid_citations"] == [4]


def test_turkish_formats_are_supported_but_question_numbers_warn():
    result = verify_answer(
        "2025 yılında kişi başı yeşil alan 5.8 metrekareydi; günlük yolcu 21.500.",
        EVIDENCE,
        question="2025 yılında kişi başı yeşil alan ne kadardı?",
    )

    assert result["unsupported_numbers"] == []
    assert result["question_only_numbers"] == ["2025"]


@pytest.mark.parametrize("label", ["Kanıt", "Kanit", "Evidence", "KANIT", "kAnIt", "eViDeNcE"])
@pytest.mark.parametrize("number", ["1", "4", "0"])
def test_citation_aliases_share_parsing_and_number_removal(label, number):
    result = verify_answer(f"Değer 22 [ \t{label}\n{number} \t].", [{"text": "Değer 22."}])
    assert result == {
        "citations": [1] if number == "1" else [],
        "invalid_citations": [] if number == "1" else [int(number)],
        "unsupported_numbers": [], "question_only_numbers": [],
    }


def test_citation_occurrences_are_deduplicated_and_sorted():
    result = verify_answer("[Evidence 4] [Kanit 2] [Kanıt 1] [Evidence 2] [Kanıt 4]", EVIDENCE)
    assert result["citations"] == [1, 2]
    assert result["invalid_citations"] == [4]
    assert result["unsupported_numbers"] == []


@pytest.mark.parametrize("text", ["[1]", "Evidence 1", "Kanıt 1", "[Evidence1]", "[Evidence -1]"])
def test_non_protocol_references_are_not_citations(text):
    result = verify_answer(text, [{"text": "Sayı yok."}])
    assert result["citations"] == []
    assert result["invalid_citations"] == []
    assert result["unsupported_numbers"] == ["1"]


@pytest.mark.parametrize("answer,evidence", [
    ("%22'ye", "yüzde 22"), ("18.000", "18000"), ("18000", "18.000"),
    ("7,4", "7.4"), ("7.4", "7,4"),
    ("1,234,567", "1.234.567"), ("1.234.567", "1234567"),
    ("1234567", "1,234,567"), ("5.800", "5800"),
])
def test_documented_number_equivalences(answer, evidence):
    result = verify_answer(answer, [{"text": evidence}])
    assert result["unsupported_numbers"] == []
    assert result["question_only_numbers"] == []


@pytest.mark.parametrize("answer,evidence", [
    ("5.800", "5,8"), ("5,8", "5.800"), ("1,234", "1234"),
    ("5,800", "5800"), ("12,34,567", "1234567"),
    ("1.234,56", "1,234.56"), ("12:30", "12.30"),
])
def test_ambiguous_or_distinct_number_forms_do_not_collapse(answer, evidence):
    assert verify_answer(answer, [{"text": evidence}])["unsupported_numbers"] == [answer]


def test_question_only_unsupported_and_evidence_numbers_stay_separate():
    result = verify_answer("2025, 2026, 2025 ve 2024", [{"text": "Yıl 2024."}], "2025 veya 2024?")
    assert result["question_only_numbers"] == ["2025"]
    assert result["unsupported_numbers"] == ["2026"]


def test_normalized_question_only_numbers_keep_answer_spelling_and_order():
    result = verify_answer("18.000 ve 7.4 ve 18.000", [{"text": "Sayı yok."}], "18000 ve 7,4?")
    assert result["question_only_numbers"] == ["18.000", "7.4"]
    assert result["unsupported_numbers"] == []


def test_number_in_question_and_evidence_is_not_question_only():
    result = verify_answer("2025", [{"text": "2025"}], "2025?")
    assert result["question_only_numbers"] == []
    assert result["unsupported_numbers"] == []


def test_existing_but_wrong_source_number_does_not_claim_semantic_support():
    result = verify_answer("Değer 22. [Kanıt 2]", [{"text": "Değer 22."}, {"text": "İlgisiz metin."}])
    # Membership in the union of evidence texts is not a claim-to-source check.
    assert result == {
        "citations": [2], "invalid_citations": [],
        "unsupported_numbers": [], "question_only_numbers": [],
    }


CASES = json.loads((Path(__file__).parent / "fixtures/answer_contract/cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", [case for case in CASES if "verification" in case], ids=lambda case: case["id"])
def test_answer_contract_fixtures(case):
    assert verify_answer(case["answer"], [{"text": case["evidence"]}], case.get("question", "")) == case["verification"]


def test_calculate_evidence_confidence_grades_and_penalties():
    from app.rag.grounding import calculate_evidence_confidence

    high_ev = [
        {"text": "Su kaybı %22.", "score": 0.92},
        {"text": "Tramvay 21.500 yolcu.", "score": 0.88},
    ]
    ver_high = verify_answer("Su kaybı %22 [Kanıt 1] ve 21500 yolcu [Kanıt 2].", high_ev)
    metrics_high = calculate_evidence_confidence(high_ev, ver_high)
    assert metrics_high["grade"] == "HIGH"
    assert metrics_high["citation_coverage_ratio"] == 1.0
    assert metrics_high["cited_source_count"] == 2
    assert metrics_high["confidence_score"] >= 0.8

    ver_penalized = verify_answer("Su kaybı %99 [Kanıt 1] [Kanıt 5].", high_ev)
    metrics_penalized = calculate_evidence_confidence(high_ev, ver_penalized)
    assert metrics_penalized["confidence_score"] < metrics_high["confidence_score"]
    assert metrics_penalized["grade"] in ("MODERATE", "LOW")

    metrics_empty = calculate_evidence_confidence([], None)
    assert metrics_empty["grade"] == "INSUFFICIENT"
    assert metrics_empty["confidence_score"] == 0.0


def test_expand_query_terms_adds_domain_synonyms():
    from app.rag.grounding import expand_query_terms

    expanded = expand_query_terms("Su ve ulaşım bütçesi ne kadar?")
    assert "su" in expanded["base_terms"]
    assert "ulaşım" in expanded["base_terms"]
    assert "tramvay" in expanded["expanded_terms"]
    assert "içme suyu" in expanded["expanded_terms"]
    assert "ödenek" in expanded["expanded_terms"]

    empty = expand_query_terms("   ")
    assert empty["base_terms"] == []
    assert empty["expanded_terms"] == []


def test_extract_key_claims_and_claim_verification_report():
    from app.rag.grounding import claim_verification_report, extract_key_claims

    answer = (
        "Su kaybı %31'den %22'ye düştü [Kanıt 1]; "
        "tramvay günde 21.500 yolcu taşıyor [Kanıt 2]. "
        "Yeni metro hattı 999 milyon liraya tamamlandı [Kanıt 1]."
    )
    claims = extract_key_claims(answer)
    assert len(claims) == 3
    assert claims[0]["citations"] == [1]
    assert claims[0]["numbers"] == ["31", "22"]
    assert claims[1]["citations"] == [2]
    assert claims[1]["numbers"] == ["21.500"]

    report = claim_verification_report(answer, EVIDENCE)
    assert report["total_claims"] == 3
    assert report["supported_claims"] == 2
    assert report["claims"][0]["supported"] is True
    assert report["claims"][1]["supported"] is True
    assert report["claims"][2]["supported"] is False
    assert report["claims"][2]["unsupported_numbers"] == ["999"]


def test_detect_evidence_conflicts_numeric_polarity_and_consensus():
    from app.rag.grounding import detect_evidence_conflicts

    consistent_chunks = [
        {"text": "Tramvay hattı günde 21.500 yolcu taşıyor."},
        {"text": "Yeşil alan miktarı kişi başı 5,8 metrekare olarak ölçüldü."},
    ]
    clean_report = detect_evidence_conflicts(consistent_chunks)
    assert clean_report["has_conflicts"] is False
    assert clean_report["conflict_count"] == 0
    assert clean_report["consensus_score"] == 1.0

    conflicting_chunks = [
        {"text": "Tramvay projesi bütçesi 86 milyon lira oldu ve yolcu sayısı arttı."},
        {"text": "Tramvay projesi bütçesi 112 milyon lira oldu ve yolcu sayısı azaldı."},
    ]
    conflict_report = detect_evidence_conflicts(
        conflicting_chunks,
        answer="Tramvay projesi bütçesi 150 milyon lira oldu [Kanıt 1].",
    )
    assert conflict_report["has_conflicts"] is True
    assert conflict_report["conflict_count"] >= 2
    conflict_types = {c["type"] for c in conflict_report["conflicts"]}
    assert "NUMERIC_DISCREPANCY" in conflict_types
    assert "POLARITY_CONTRADICTION" in conflict_types
    assert "ANSWER_EVIDENCE_MISMATCH" in conflict_types
    assert conflict_report["consensus_score"] < 1.0


def test_score_evidence_freshness_and_authority():
    from app.rag.grounding import score_evidence_freshness_and_authority

    empty_report = score_evidence_freshness_and_authority([], reference_year=2026)
    assert empty_report["trust_grade"] == "INSUFFICIENT"
    assert empty_report["combined_trust_score"] == 0.0
    assert empty_report["items"] == []

    sources = [
        {
            "filename": "mavi_korfez_2025.md",
            "source": "https://mavikorfez.bel.tr/faaliyet-raporu-2025",
            "text": "Mavi Körfez Belediyesi 2025 yılı faaliyet raporu resmi bütçe verileri.",
        },
        {
            "filename": "eski_forum_yorumu.txt",
            "source_type": "blog",
            "published_year": 2019,
            "text": "2019 yılında forumda paylaşılan doğrulanmamış yorum.",
        },
        {
            "text": "Kurumsal altyapı denetim raporu özetidir.",
            "metadata": {
                "filename": "denetim.pdf",
                "uploaded_at": "2024-05-12T10:00:00+00:00",
                "ocr": True,
            },
        },
    ]

    report = score_evidence_freshness_and_authority(sources, reference_year=2026)
    assert report["reference_year"] == 2026
    assert report["stale_source_count"] == 1
    assert len(report["items"]) == 3

    official_item, stale_item, ocr_item = report["items"]
    assert official_item["authority_tier"] == "OFFICIAL"
    assert official_item["freshness_label"] == "CURRENT"
    assert official_item["detected_year"] == 2025
    assert official_item["trust_score"] > 0.85

    assert stale_item["authority_tier"] == "SECONDARY"
    assert stale_item["freshness_label"] == "STALE"
    assert stale_item["age_years"] == 7
    assert stale_item["trust_score"] < official_item["trust_score"]

    assert ocr_item["authority_tier"] == "INSTITUTIONAL"
    assert ocr_item["detected_year"] == 2024
    assert ocr_item["authority_score"] == 0.77
    assert report["trust_grade"] in ("MODERATE", "HIGH")


def test_build_citation_coverage_matrix():
    from app.rag.grounding import build_citation_coverage_matrix

    sources = [
        {"text": "Su kayıp oranı yüzde 31'den yüzde 22'ye düştü."},
        {"text": "Tramvay hattı günde 21.500 yolcu taşıyor."},
        {"text": "Sahil parkına 1.150 ağaç dikildi."},
    ]

    empty_matrix = build_citation_coverage_matrix("   ", sources)
    assert empty_matrix["sentence_count"] == 0
    assert empty_matrix["unused_evidence_indices"] == [1, 2, 3]
    assert empty_matrix["matrix"] == []

    answer = (
        "Su kayıp oranı yüzde 22 seviyesine düştü [Kanıt 1]. "
        "Belediye ek olarak yeni duraklar açmayı planlıyor. "
        "Tramvay günde 21.500 yolcu taşıyor [Kanıt 2] [Kanıt 5]."
    )
    cov = build_citation_coverage_matrix(answer, sources)

    assert cov["sentence_count"] == 3
    assert cov["cited_sentence_count"] == 2
    assert cov["uncited_sentence_count"] == 1
    assert cov["sentence_citation_coverage_ratio"] == 0.6667
    assert cov["used_evidence_indices"] == [1, 2]
    assert cov["unused_evidence_indices"] == [3]
    assert cov["evidence_utilization_ratio"] == 0.6667
    assert cov["orphan_citations"] == [5]

    row1, row2, row3 = cov["matrix"]
    assert row1["coverage_status"] == "CITED"
    assert row1["source_vector"] == [True, False, False]
    assert row1["evidence_links"][1]["cited"] is True
    assert row1["evidence_links"][1]["alignment_score"] > row1["evidence_links"][2]["alignment_score"]

    assert row2["coverage_status"] == "UNCITED"
    assert row2["is_cited"] is False
    assert row2["source_vector"] == [False, False, False]

    assert row3["coverage_status"] == "CITED_WITH_INVALID"
    assert row3["cited_evidence"] == [2]
    assert row3["invalid_citations"] == [5]
    assert row3["evidence_links"][2]["alignment_score"] > 0.5

