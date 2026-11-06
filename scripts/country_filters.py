"""Deterministic country filters on explicit markers (WWW replication across countries, 2026-11-06).

Canada is the audited strict_canada filter, unchanged. Every other filter follows the same recipe:
country name and demonym, major cities or regions, national cyber/privacy agencies, flag emoji and
national domain. Some markers are case-sensitive abbreviations (US, USA, UK). Precision is audited with
LLM labels (www_country_audit.py), never presented as human validation.
"""
import re

from strict_canada_filter import strict_canada

I, S = re.I, 0
SPEC = {
    "united_states": ([r"united states", r"\bamerica(n|ns)?\b", r"\bu\.s\.(a\.)?", r"washington,? d\.?c\.?",
                       r"\bcalifornia\b", r"\btexas\b", r"\bnew york\b", r"\bflorida\b", r"\billinois\b", r"\bohio\b",
                       r"\bcisa\b", r"\bfbi\b", r"\bnsa\b", r"\bftc\b", r"\bhhs\b", r"\bdhs\b", r"\bdoj\b",
                       r"department of (justice|homeland security|defense)", r"\bpentagon\b", r"white house",
                       r"\bcongress(ional)?\b", r"[\w-]+\.gov\b(?!\.)"],
                      [r"\bUS\b", r"\bUSA\b"], "\U0001F1FA\U0001F1F8"),
    "united_kingdom": ([r"united kingdom", r"\bbritain\b", r"\bbritish\b(?! columbia)", r"\bengland\b", r"\bscotland\b",
                        r"\bwales\b(?<!new south wales)", r"northern ireland", r"\blondon\b(?!,? on)", r"\bmanchester\b",
                        r"\bbirmingham\b", r"\bedinburgh\b", r"\bglasgow\b", r"ncsc\.gov\.uk", r"\bnhs\b",
                        r"information commissioner'?s office", r"[\w-]+\.uk\b"],
                       [r"\bUK\b"], "\U0001F1EC\U0001F1E7"),
    "australia": ([r"\baustralia(n|ns)?\b", r"\baussie\b", r"\bsydney\b", r"\bmelbourne\b", r"\bbrisbane\b", r"\bperth\b",
                   r"\badelaide\b", r"\bcanberra\b", r"\bqueensland\b", r"new south wales", r"\bnsw\b", r"\bacsc\b",
                   r"\boaic\b", r"\bmedibank\b", r"\boptus\b", r"[\w-]+\.au\b"],
                  [], "\U0001F1E6\U0001F1FA"),
    "india": ([r"\bindia\b", r"\bindian\b(?! ocean)", r"\bdelhi\b", r"\bmumbai\b", r"\bbengaluru\b", r"\bbangalore\b",
               r"\bhyderabad\b", r"\bchennai\b", r"\bkolkata\b", r"\bpune\b", r"\bcert-in\b", r"\bmeity\b", r"\baadhaar\b",
               r"\brupees?\b", r"\b(?!lnkd\.in\b)(?!linked\.in\b)(?!buff\.ly)[\w-]+\.in\b(?!\.)"],
              [], "\U0001F1EE\U0001F1F3"),
    "germany": ([r"\bgermany\b", r"\bgerman\b", r"\bdeutschland\b", r"\bberlin\b", r"\bmunich\b", r"m[üu]nchen", r"\bhamburg\b",
                 r"\bfrankfurt\b", r"\bcologne\b", r"k[öo]ln\b", r"\bbundes\w+", r"[\w-]+\.de\b(?!\.)"],
                [r"\bBSI\b"], "\U0001F1E9\U0001F1EA"),
    "france": ([r"\bfrance\b", r"\bfrench\b", r"\bparis\b", r"\blyon\b", r"\bmarseille\b", r"\btoulouse\b", r"\banssi\b",
                r"\bcnil\b", r"[\w-]+\.fr\b(?!\.)"],
               [], "\U0001F1EB\U0001F1F7"),
    "japan": ([r"\bjapan(ese)?\b", r"\btokyo\b", r"\bosaka\b", r"\bkyoto\b", r"\byokohama\b", r"\bjpcert\b", r"\bnisc\b",
               r"[\w-]+\.jp\b(?!\.)"],
              [], "\U0001F1EF\U0001F1F5"),
    "netherlands": ([r"\bnetherlands\b", r"\bdutch\b", r"\bholland\b", r"\bamsterdam\b", r"\brotterdam\b", r"the hague",
                     r"\butrecht\b", r"\beindhoven\b", r"ncsc-nl", r"[\w-]+\.nl\b(?!\.)"],
                    [], "\U0001F1F3\U0001F1F1"),
    "singapore": ([r"\bsingapore(an|ans)?\b", r"\bpdpc\b", r"[\w-]+\.sg\b(?!\.)"],
                  [], "\U0001F1F8\U0001F1EC"),
}
_PAT = {k: (re.compile("|".join(ci), I), re.compile("|".join(cs)) if cs else None, flag) for k, (ci, cs, flag) in SPEC.items()}
COUNTRIES = ["canada"] + list(SPEC)


def countries(text):
    t = text or ""
    out = ["canada"] if strict_canada(t) else []
    for k, (ci, cs, flag) in _PAT.items():
        if ci.search(t) or (cs is not None and cs.search(t)) or flag in t:
            out.append(k)
    return out
