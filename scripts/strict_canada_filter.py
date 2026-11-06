"""Deterministic strict Canada filter (explicit Canadian markers only). Validated 2026-11-05 against
annotator A on the 150-item Canada audit: precision 0.979 (46/47), recall 0.902 (46/51)."""
import re
TERMS = [r"canad(a|ian|iens?|iennes?|ienne)", r"qu[ée]b[ée]c(ois|oise)?", r"ontario", r"alberta", r"british columbia",
         r"manitoba", r"saskatchewan", r"nova scotia", r"new brunswick", r"newfoundland", r"prince edward island",
         r"yukon", r"nunavut", r"northwest territories", r"toronto", r"montr[ée]al", r"vancouver", r"calgary",
         r"edmonton", r"ottawa", r"winnipeg", r"halifax", r"saskatoon", r"regina", r"victoria,? b\.?c\.?",
         r"hamilton,? on", r"mississauga", r"brampton", r"laval", r"gatineau", r"sherbrooke", r"kelowna",
         r"\bcccs\b", r"cyber centre", r"\brcmp\b", r"hydro[- ]qu[ée]bec", r"\bcbc\b", r"radio[- ]canada",
         r"desjardins", r"\bunb\b", r"university of (toronto|waterloo|british columbia|calgary|alberta|manitoba)",
         r"mcgill", r"waterloo", r"\bccq\b", r"privacy commissioner of canada", r"\bopc\b"]
PAT = re.compile("|".join(TERMS), re.I)
FLAG = "\U0001F1E8\U0001F1E6"
DOM = re.compile(r"\b[\w-]+\.ca\b(?!\w)", re.I)
def strict_canada(text):
    t = text or ""
    return bool(PAT.search(t) or FLAG in t or DOM.search(t))
