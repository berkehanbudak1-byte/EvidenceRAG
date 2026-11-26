import re


_CITATION = re.compile(r"\[\s*(?:Kan[ıi]t|Evidence)\s+([0-9]+)\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_DOT_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
_COMMA_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3}){2,}")
_DECIMAL = re.compile(r"\d+[.,]\d+")


def verify_answer(answer, evidence, question=""):
    """Textual warnings, not semantic verification or a reason to block an answer.

    Citation membership and number presence across all evidence do not prove that
    the cited source supports the claim. Question-only numbers warn separately.
    """
    cited = set()

    def remove_citation(match):
        # One parser both records citations and excludes their complete markers
        # from number scanning, including out-of-range citation numbers.
        cited.add(int(match.group(1)))
        return " "

    number_text = _CITATION.sub(remove_citation, answer)
    cited = sorted(cited)
    valid = set(range(1, len(evidence) + 1))

    evidence_numbers = {
        _normalize_number(number)
        for item in evidence
        for number in _NUMBER.findall(item["text"])
    }
    question_numbers = {_normalize_number(number) for number in _NUMBER.findall(question)}
    unsupported_numbers = []
    question_only_numbers = []
    for number in _NUMBER.findall(number_text):
        normalized = _normalize_number(number)
        if normalized in evidence_numbers:
            continue
        findings = question_only_numbers if normalized in question_numbers else unsupported_numbers
        if number not in findings:
            findings.append(number)

    return {
        "citations": [number for number in cited if number in valid],
        "invalid_citations": [number for number in cited if number not in valid],
        "unsupported_numbers": unsupported_numbers,
        "question_only_numbers": question_only_numbers,
    }


def _normalize_number(number):
    """Conservative textual locale rules, documented in README's answer contract.

    A single comma is decimal; a single dot with a valid three-digit group is
    Turkish thousands. Multiple uniform three-digit groups are unambiguous.
    Mixed/malformed separators and time colons are kept, not guessed.
    """
    if _DOT_THOUSANDS.fullmatch(number):
        return number.replace(".", "")
    if _COMMA_THOUSANDS.fullmatch(number):
        return number.replace(",", "")
    if _DECIMAL.fullmatch(number):
        return number.replace(",", ".")
    return number


def calculate_evidence_confidence(evidence, verification):
    """Calculate an explicit evidence density and citation coverage confidence score.

    Combines citation coverage ratio, average retrieval score of cited sources,
    and penalties for invalid citations or unsupported numbers.
    """
    if not evidence or not verification:
        return {
            "confidence_score": 0.0,
            "citation_coverage_ratio": 0.0,
            "cited_source_count": 0,
            "total_source_count": len(evidence) if evidence else 0,
            "grade": "INSUFFICIENT",
        }

    total_sources = len(evidence)
    valid_citations = verification.get("citations", [])
    invalid_citations = verification.get("invalid_citations", [])
    unsupported_numbers = verification.get("unsupported_numbers", [])

    cited_count = len(valid_citations)
    coverage_ratio = round(cited_count / total_sources, 4) if total_sources > 0 else 0.0

    cited_scores = [
        float(evidence[idx - 1].get("score", 0.75))
        for idx in valid_citations
        if 1 <= idx <= total_sources and evidence[idx - 1].get("score") is not None
    ]
    avg_retrieval_score = (
        sum(cited_scores) / len(cited_scores)
        if cited_scores
        else (0.75 if cited_count > 0 else 0.0)
    )

    base_score = (0.55 * min(1.0, avg_retrieval_score)) + (0.45 * min(1.0, coverage_ratio * 1.25))
    penalty = (0.2 * len(invalid_citations)) + (0.15 * len(unsupported_numbers))
    final_score = round(max(0.0, min(1.0, base_score - penalty)), 4) if cited_count > 0 else 0.0

    if final_score >= 0.75 and not invalid_citations and not unsupported_numbers:
        grade = "HIGH"
    elif final_score >= 0.45:
        grade = "MODERATE"
    elif final_score > 0.0:
        grade = "LOW"
    else:
        grade = "UNVERIFIED"

    return {
        "confidence_score": final_score,
        "citation_coverage_ratio": coverage_ratio,
        "cited_source_count": cited_count,
        "total_source_count": total_sources,
        "grade": grade,
    }


_TURKISH_SYNONYMS = {
    "bütçe": ["ödenek", "harcama", "maliyet", "finansman"],
    "maliyet": ["bütçe", "tutar", "harcama"],
    "ulaşım": ["tramvay", "otobüs", "toplu taşıma", "hat", "yolcu"],
    "tramvay": ["raylı sistem", "ulaşım", "yolcu"],
    "su": ["içme suyu", "şebeke", "kayıp-kaçak", "altyapı"],
    "yeşil": ["park", "ağaçlandırma", "rekreasyon", "metrekare"],
    "atık": ["geri dönüşüm", "çöp", "bertaraf"],
}

_STOPWORDS = {
    "ve", "ile", "için", "bir", "bu", "şu", "ne", "kadar", "mı", "mi", "mu", "mü",
    "da", "de", "ta", "te", "göre", "olan", "olarak", "daha", "çok", "en",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*;\s*")
_WORD_TOKEN = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")


def expand_query_terms(query):
    """Expand a user query into normalized search tokens and domain synonyms."""
    if not query or not query.strip():
        return {"original_query": query or "", "base_terms": [], "expanded_terms": []}

    raw_tokens = [tok.lower() for tok in _WORD_TOKEN.findall(query)]
    base_terms = []
    for tok in raw_tokens:
        if len(tok) >= 2 and tok not in _STOPWORDS and tok not in base_terms:
            base_terms.append(tok)

    expanded = list(base_terms)
    for term in base_terms:
        for key, syns in _TURKISH_SYNONYMS.items():
            if term == key or term.startswith(key):
                for syn in syns:
                    if syn not in expanded:
                        expanded.append(syn)

    return {
        "original_query": query,
        "base_terms": base_terms,
        "expanded_terms": expanded,
    }


def extract_key_claims(answer):
    """Extract verifiable factual claims (sentences with numbers, citations, or entities) from an answer."""
    if not answer or not answer.strip():
        return []

    raw_clauses = [seg.strip() for seg in _SENTENCE_SPLIT.split(answer.strip()) if seg.strip()]
    claims = []
    for idx, clause in enumerate(raw_clauses, start=1):
        citations = sorted({int(m.group(1)) for m in _CITATION.finditer(clause)})
        clean_text = _CITATION.sub(" ", clause).strip()
        clean_text = re.sub(r"\s+", " ", clean_text).strip(" .;")
        numbers = []
        for num in _NUMBER.findall(clean_text):
            if num not in numbers:
                numbers.append(num)

        tokens = [
            tok.lower()
            for tok in _WORD_TOKEN.findall(clean_text)
            if len(tok) >= 3 and tok.lower() not in _STOPWORDS and not tok.isdigit()
        ]
        if not clean_text:
            continue
        claims.append({
            "claim_id": idx,
            "text": clean_text,
            "citations": citations,
            "numbers": numbers,
            "key_terms": tokens,
        })
    return claims


def claim_verification_report(answer, evidence):
    """Verify per-claim evidence support across extracted claims."""
    claims = extract_key_claims(answer)
    if not claims:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "support_ratio": 0.0,
            "claims": [],
        }

    total_evidence = len(evidence) if evidence else 0
    all_evidence_numbers = [
        {_normalize_number(n) for n in _NUMBER.findall(item.get("text", ""))}
        for item in (evidence or [])
    ]
    all_evidence_texts = [item.get("text", "").lower() for item in (evidence or [])]

    evaluated_claims = []
    supported_count = 0

    for claim in claims:
        valid_cits = [c for c in claim["citations"] if 1 <= c <= total_evidence]
        candidate_indices = [c - 1 for c in valid_cits] if valid_cits else list(range(total_evidence))

        # Check numeric match in candidate evidence sources
        norm_nums = [_normalize_number(n) for n in claim["numbers"]]
        missing_nums = []
        for raw_n, norm_n in zip(claim["numbers"], norm_nums):
            if not any(norm_n in all_evidence_numbers[i] for i in candidate_indices):
                missing_nums.append(raw_n)

        # Check key term overlap in candidate evidence sources
        matched_terms = [
            term
            for term in claim["key_terms"]
            if any(term[:4] in all_evidence_texts[i] for i in candidate_indices)
        ]
        term_overlap = (
            round(len(matched_terms) / len(claim["key_terms"]), 4)
            if claim["key_terms"]
            else 1.0
        )

        is_supported = (
            total_evidence > 0
            and len(missing_nums) == 0
            and (len(claim["numbers"]) > 0 or term_overlap >= 0.4)
            and (len(claim["citations"]) == 0 or len(valid_cits) == len(claim["citations"]))
        )
        if is_supported:
            supported_count += 1

        evaluated_claims.append({
            "claim_id": claim["claim_id"],
            "text": claim["text"],
            "citations": claim["citations"],
            "numbers": claim["numbers"],
            "unsupported_numbers": missing_nums,
            "term_overlap_ratio": term_overlap,
            "supported": is_supported,
        })

    return {
        "total_claims": len(evaluated_claims),
        "supported_claims": supported_count,
        "support_ratio": round(supported_count / len(evaluated_claims), 4),
        "claims": evaluated_claims,
    }

