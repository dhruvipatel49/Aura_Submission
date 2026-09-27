#!/usr/bin/env python3
"""
Text normalization utilities for business entity resolution.
Optimized for speed: pre-compiled regexes, token-based replacements.
"""

import re
import unicodedata
from functools import lru_cache

# ---- Pre-compiled regex patterns ----
RE_BRACKETS = re.compile(r'[\[\(].*?[\]\)]')
RE_PUNCT = re.compile(r'[^\w\s]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_DIGITS = re.compile(r'\d+')

# Legal suffixes as a set for fast token-level removal
LEGAL_SUFFIX_TOKENS = {
    # US / India / UK
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'llc', 'llp', 'ltd', 'limited', 'pvt', 'private', 'pllc', 'plc',
    'pc', 'pa', 'enterprises', 'enterprise', 'services', 'service',
    'partners', 'partner', 'associates', 'associate', 'solutions',
    'solution', 'group', 'holdings', 'holding', 'center', 'centre',
    # France legal entity forms
    'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'snc', 'gie', 'sca',
    'scs', 'selarl', 'scp', 'ei', 'eirl', 'microentreprise', 'societe',
}

# Multi-word suffixes to strip (as tuples for endswith check)
LEGAL_SUFFIX_PAIRS = [
    ('private', 'limited'), ('pvt', 'ltd'), ('pvt', 'limited'),
    ('societe', 'anonyme'), ('par', 'actions'), ('actions', 'simplifiee'),
    ('responsabilite', 'limitee'),
]

# Address abbreviation map (token-level replacement, both directions → canonical)
ADDR_TOKEN_MAP = {
    # English
    'street': 'st', 'avenue': 'ave', 'road': 'rd', 'drive': 'dr',
    'boulevard': 'blvd', 'lane': 'ln', 'court': 'ct', 'place': 'pl',
    'circle': 'cir', 'terrace': 'ter', 'highway': 'hwy', 'parkway': 'pkwy',
    'trail': 'trl', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'apartment': 'apt', 'suite': 'ste', 'building': 'bldg', 'floor': 'fl',
    'number': 'no',
    # French
    'rue': 'st', 'av': 'ave', 'bd': 'blvd', 'chemin': 'ch', 'route': 'rd',
    'impasse': 'imp', 'allee': 'all', 'cours': 'crs', 'passage': 'pass',
    'faubourg': 'fg', 'etage': 'fl', 'batiment': 'bldg', 'appartement': 'apt',
}

# State name → abbreviation (token-level, only applied to individual tokens or pairs)
_US_STATES_SINGLE = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne',
    'nevada': 'nv', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or',
    'pennsylvania': 'pa', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
    'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'wisconsin': 'wi', 'wyoming': 'wy',
    # Indian single-word states
    'karnataka': 'ka', 'kerala': 'kl', 'maharashtra': 'mh',
    'rajasthan': 'rj', 'goa': 'ga', 'gujarat': 'gj',
    'haryana': 'hr', 'jharkhand': 'jh', 'bihar': 'br',
    'chhattisgarh': 'cg', 'manipur': 'mn', 'meghalaya': 'ml',
    'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od', 'orissa': 'od',
    'punjab': 'pb', 'sikkim': 'sk', 'telangana': 'tg', 'tripura': 'tr',
    'uttarakhand': 'uk', 'delhi': 'dl', 'chandigarh': 'ch',
    'puducherry': 'py', 'pondicherry': 'py', 'ladakh': 'la',
}

# Two-word states: (word1, word2) -> abbr
_STATES_TWO_WORD = {
    ('new', 'hampshire'): 'nh', ('new', 'jersey'): 'nj',
    ('new', 'mexico'): 'nm', ('new', 'york'): 'ny', ('new', 'delhi'): 'dl',
    ('north', 'carolina'): 'nc', ('north', 'dakota'): 'nd',
    ('south', 'carolina'): 'sc', ('south', 'dakota'): 'sd',
    ('west', 'virginia'): 'wv', ('rhode', 'island'): 'ri',
    ('tamil', 'nadu'): 'tn', ('andhra', 'pradesh'): 'ap',
    ('arunachal', 'pradesh'): 'ar', ('himachal', 'pradesh'): 'hp',
    ('madhya', 'pradesh'): 'mp', ('uttar', 'pradesh'): 'up',
    ('west', 'bengal'): 'wb', ('west', 'delhi'): 'dl',
    ('south', 'delhi'): 'dl', ('north', 'delhi'): 'dl',
    ('east', 'delhi'): 'dl', ('central', 'delhi'): 'dl',
}


def strip_accents(s):
    """Remove accents/diacritics and expand ligatures, keeping base characters."""
    if not s:
        return s
    s = s.replace('œ', 'oe').replace('æ', 'ae').replace('Œ', 'oe').replace('Æ', 'ae')
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def is_non_latin(text):
    """Check if text contains non-Latin script characters."""
    if not text:
        return False
    for c in text:
        if c.isalpha() and ord(c) > 127:
            # Quick check: if char code > 127 and it's alphabetic, likely non-Latin
            name = unicodedata.name(c, '')
            if any(s in name for s in ('DEVANAGARI', 'TAMIL', 'TELUGU', 'KANNADA',
                                        'BENGALI', 'GUJARATI', 'MALAYALAM', 'GURMUKHI',
                                        'ORIYA', 'ARABIC', 'CJK')):
                return True
    return False


def normalize_name(name):
    """
    Fast business name normalization:
    - lowercase, strip accents, remove brackets/punct
    - remove legal suffix tokens
    - collapse whitespace
    """
    if not name or not isinstance(name, str):
        return ''

    s = name.lower()
    s = strip_accents(s)
    s = RE_BRACKETS.sub(' ', s)
    s = s.replace('&', ' and ').replace('+', ' and ')
    s = s.replace('.', '')  # FIX 1: Strip periods for acronyms BEFORE replacing other punct with space
    s = RE_PUNCT.sub(' ', s)

    # Token-level processing: remove legal suffixes
    tokens = s.split()
    cleaned = [t for t in tokens if t not in LEGAL_SUFFIX_TOKENS and len(t) > 0]

    # Remove multi-word suffix pairs from end
    for w1, w2 in LEGAL_SUFFIX_PAIRS:
        if len(cleaned) >= 2 and cleaned[-2] == w1 and cleaned[-1] == w2:
            cleaned = cleaned[:-2]

    # FIX 2: If stripping suffixes left the name completely empty (e.g. "Urology Partners PLLC" -> "urology" which might be skipped later, but if it was just "Partners PLLC" -> ""), keep original tokens
    if not cleaned and tokens:
        cleaned = tokens

    result = ' '.join(cleaned).strip()
    return result


def normalize_name_sorted_tokens(name):
    """Return sorted unique tokens of normalized name."""
    norm = normalize_name(name)
    if not norm:
        return ''
    return ' '.join(sorted(set(norm.split())))


def normalize_address(addr):
    """
    Fast address normalization using token-level replacements.
    No per-string regex loops over state/abbreviation lists.
    """
    if not addr or not isinstance(addr, str):
        return ''

    s = addr.lower()
    # Remove null markers
    s = s.replace('<null>', ' ').replace('null', ' ')
    s = strip_accents(s)
    s = s.replace('&', ' and ')
    s = RE_PUNCT.sub(' ', s)

    # Token-level processing
    tokens = s.split()
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Check two-word state names first
        if i + 1 < len(tokens):
            pair = (tok, tokens[i + 1])
            if pair in _STATES_TWO_WORD:
                result.append(_STATES_TWO_WORD[pair])
                i += 2
                continue

        # Single-word state
        if tok in _US_STATES_SINGLE:
            result.append(_US_STATES_SINGLE[tok])
        # Address abbreviation
        elif tok in ADDR_TOKEN_MAP:
            result.append(ADDR_TOKEN_MAP[tok])
        else:
            result.append(tok)
        i += 1

    return ' '.join(result)


def extract_digits(s):
    """Extract all digit sequences from a string."""
    if not s or not isinstance(s, str):
        return []
    return RE_DIGITS.findall(s)


def get_name_tokens(name):
    """Get set of normalized tokens from a name."""
    norm = normalize_name(name)
    if not norm:
        return set()
    return set(norm.split())


def get_first_n_chars(name, n=5):
    """Get first n characters of normalized name."""
    norm = normalize_name(name)
    return norm[:n] if norm else ''
