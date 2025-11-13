# ml/parser.py
import re
from typing import List, Dict, Any
import spacy
from dateutil import parser as dateparser

# load spaCy model (ensure en_core_web_sm is installed)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError("Please install spaCy model: python -m spacy download en_core_web_sm") from e

# small skills seed list (extend this file or load from disk)
SKILLS_SEED = [
    "python", "java", "c++", "c", "tensorflow", "pytorch", "scikit-learn",
    "docker", "kubernetes", "aws", "gcp", "azure", "nlp", "computer vision",
    "pandas", "numpy", "sql", "nosql", "react", "nodejs", "git", "rest",
]

EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"(\+?\d{1,4}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?[\d\s\-]{6,12}")

DATE_RANGE_RE = re.compile(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|'
                           r'January|February|March|April|May|June|July|August|September|October|November|December|\d{4})'
                           r'[^,\n\r]{0,30}(?:\d{4}))', re.IGNORECASE)

SECTION_HEADERS = ["experience", "work experience", "professional experience",
                   "education", "skills", "projects", "certifications", "summary", "objective"]


def extract_emails(text: str) -> List[str]:
    return list(set(m.group(0) for m in EMAIL_RE.finditer(text)))


def extract_phones(text: str) -> List[str]:
    phones = []
    for m in PHONE_RE.finditer(text):
        phones.append(m.group(0).strip())
    # simple normalization: unique
    return list(dict.fromkeys(phones))


def extract_skills(text: str, skills_seed: List[str] = SKILLS_SEED) -> List[str]:
    text_low = text.lower()
    found = set()
    for s in skills_seed:
        if s.lower() in text_low:
            found.add(s)
    return sorted(found)


def split_into_sections(text: str) -> Dict[str, str]:
    """
    Naive section splitter: looks for header keywords and splits. Returns dict header->block.
    """
    lines = [l.strip() for l in text.splitlines()]
    # create simple positions for headers
    sections = {}
    current = "header"
    buffer = []
    for l in lines:
        low = l.lower().strip(':').strip()
        if len(l) == 0:
            # preserve blank lines as separators
            if buffer:
                sections.setdefault(current, "")
                sections[current] += "\n".join(buffer) + "\n"
                buffer = []
            continue
        if any(h in low for h in SECTION_HEADERS) and len(l.split()) <= 5:
            # start a new section
            if buffer:
                sections.setdefault(current, "")
                sections[current] += "\n".join(buffer) + "\n"
                buffer = []
            current = low
            continue
        buffer.append(l)
    if buffer:
        sections.setdefault(current, "")
        sections[current] += "\n".join(buffer) + "\n"
    return sections


def extract_name(doc) -> str:
    # use spaCy PERSON entity at top of doc
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # prefer those in the first 120 chars
            if ent.start_char < 120:
                return ent.text
    # fallback: first non-empty line
    first_line = doc.text.strip().splitlines()[0]
    if len(first_line.split()) <= 4:
        return first_line.strip()
    return ""


def parse_experience_block(block_text: str) -> List[Dict[str, Any]]:
    """
    Very lightweight parsing: split into lines, try to find lines with dates and titles.
    """
    out = []
    lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        # detect date in same line
        dates = DATE_RANGE_RE.findall(line)
        if dates:
            # try to parse title from beginning of line before dates
            title = line.split(' - ')[0]
            date_text = dates[0]
            # fallback to next line as description
            bullets = []
            j = i + 1
            while j < len(lines) and len(lines[j].split()) > 2:
                # collect bullets until blank line or next date
                if DATE_RANGE_RE.search(lines[j]):
                    break
                bullets.append(lines[j])
                j += 1
            out.append({
                "title": title.strip(),
                "date_text": date_text.strip(),
                "bullets": bullets
            })
            i = j
            continue
        # else try to lookahead for a date in next line
        if i + 1 < len(lines) and DATE_RANGE_RE.search(lines[i + 1]):
            title = line
            date_text = DATE_RANGE_RE.search(lines[i + 1]).group(0)
            bullets = []
            j = i + 2
            while j < len(lines) and not DATE_RANGE_RE.search(lines[j]):
                bullets.append(lines[j])
                j += 1
            out.append({
                "title": title.strip(),
                "date_text": date_text.strip(),
                "bullets": bullets
            })
            i = j
            continue
        # otherwise, treat as short project/summary line
        out.append({
            "title": line,
            "date_text": "",
            "bullets": []
        })
        i += 1
    return out


def parse_text(text: str) -> Dict[str, Any]:
    """
    Main entry point: accepts plain text (already OCRed) and returns structured JSON.
    """
    doc = nlp(text)
    name = extract_name(doc)
    emails = extract_emails(text)
    phones = extract_phones(text)
    skills = extract_skills(text)
    sections = split_into_sections(text)

    experience_blocks = []
    # try common section keys
    for key in sections:
        if "experience" in key or "work" in key:
            experience_blocks.extend(parse_experience_block(sections[key]))

    education = []
    for key in sections:
        if "education" in key:
            # naive split into lines; each line is a degree entry
            for l in sections[key].splitlines():
                if l.strip():
                    education.append(l.strip())

    return {
        "name": name,
        "emails": emails,
        "phones": phones,
        "skills": skills,
        "experience": experience_blocks,
        "education": education,
        "sections": list(sections.keys()),
        "raw_text": text
    }


if __name__ == "__main__":
    # small demo when run directly
    sample = """Alice B
    alice@example.com | +91 9876543210

    Professional Experience
    ML Engineer - Acme Corp Feb 2019 - Sep 2022
    - Built models using python and pytorch
    - Deployed with docker and AWS

    Education
    B.Tech in Computer Science, XYZ University, 2018

    Skills
    Python, PyTorch, TensorFlow, Docker, AWS
    """
    import json
    parsed = parse_text(sample)
    print(json.dumps(parsed, indent=2))