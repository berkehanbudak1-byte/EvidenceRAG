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
