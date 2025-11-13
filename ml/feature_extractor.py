from typing import Dict, Any
from dateutil import parser as dateparser
import re
import math
from datetime import datetime

DATE_RE = re.compile(r'(\d{4})')

def estimate_years_experience(parsed: Dict[str, Any]) -> float:
    """
    Very naive years-of-experience estimator using date_text fields.
    Looks for 4-digit years in date_text and computes ranges.
    """
    years = []
    for exp in parsed.get("experience", []):
        dt = exp.get("date_text", "")
        found = DATE_RE.findall(dt)
        if len(found) >= 2:
            try:
                start = int(found[0])
                end = int(found[1])
                years.append(max(0, end - start))
            except:
                continue
        elif len(found) == 1:
            # single year provided, assume 1 year
            try:
                years.append(1)
            except:
                continue
    if not years:
        # fallback heuristic: count bullets and scale
        bullets = sum(len(exp.get("bullets", [])) for exp in parsed.get("experience", []))
        return min(1.0 + bullets * 0.5, 20.0)
    return float(sum(years))


def count_skills(parsed: Dict[str, Any]) -> int:
    return len(parsed.get("skills", []))


def formatting_score(parsed: Dict[str, Any]) -> float:
    """
    Simple heuristics: number of sections, presence of email/phone, presence of bullets.
    """
    score = 0.0
    # sections: prefer at least 3 sections
    n_sections = max(0, len(parsed.get("sections", [])))
    score += min(n_sections, 5) * 2.0
    # email/phone
    if parsed.get("emails"):
        score += 3.0
    if parsed.get("phones"):
        score += 2.0
    # bullets
    bullets = sum(len(exp.get("bullets", [])) for exp in parsed.get("experience", []))
    score += min(bullets, 10) * 0.5
    return score


def extract_features(parsed: Dict[str, Any]) -> Dict[str, float]:
    feats = {}
    feats["years_exp"] = estimate_years_experience(parsed)
    feats["skill_count"] = count_skills(parsed)
    feats["format_score"] = formatting_score(parsed)
    feats["num_experience_items"] = max(0, len(parsed.get("experience", [])))
    return feats


if __name__ == "__main__":
    # demo
    from parser import parse_text
    sample = "..."
    parsed = parse_text(sample)
    print(extract_features(parsed))