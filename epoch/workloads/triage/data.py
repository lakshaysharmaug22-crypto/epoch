"""Synthetic home-loan operations tickets (partner desk), six queues.

Deterministic and code-mixed: templates + slot values + Hinglish + typos + cross-queue context
clauses that confuse lexical matchers. `drift` uses held-out templates never seen in training,
which is what the self-healing scenario feeds in production.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

QUEUES: dict[str, dict] = {
    "disbursal": {
        "title": "Disbursal",
        "desc": "Release of sanctioned loan money: disbursement/tranche status, cheque/DD/RTGS to builder or seller, disbursal on hold.",
        "aliases": [
            "disbursement pending", "disbursal status", "tranche release", "part disbursement",
            "disbursal cheque not received", "dd handover", "rtgs to builder", "loan amount not credited",
            "disbursal on hold", "disbursal request pending",
        ],
    },
    "documents": {
        "title": "Documents & KYC",
        "desc": "Customer documents: KYC, income proofs, ITR, bank statements, property papers, OTC/PDD, checklists and re-submission.",
        "aliases": [
            "kyc verification pending", "documents required", "document checklist", "itr form 16",
            "bank statement rejected", "salary slips", "pdd otc pending", "property papers original",
            "resubmit documents", "co-applicant documents missing",
        ],
    },
    "sanction": {
        "title": "Sanction & Credit",
        "desc": "Credit decisions: approval status, sanction letter, rejection reason, eligibility, ROI/tenure/amount changes, revalidation.",
        "aliases": [
            "sanction letter not received", "loan approval status", "file rejected reason", "roi different in sanction",
            "eligibility check", "increase sanction amount", "credit decision pending", "sanction revalidation",
            "in principle approval", "tenure change in sanction",
        ],
    },
    "payout": {
        "title": "Partner Payout",
        "desc": "Partner/DSA earnings: payout or commission not credited, payout invoice, GST/TDS on payout, payout cycle and statements.",
        "aliases": [
            "payout not credited", "commission pending", "payout invoice rejected", "tds on payout",
            "payout percentage", "payout cycle", "payout statement mismatch", "gst on commission",
            "invoice format payout", "partner earnings",
        ],
    },
    "tech": {
        "title": "App & Portal",
        "desc": "Partner app/portal/CRM problems: login, OTP, password, crashes, upload errors, lead not visible, status not updating.",
        "aliases": [
            "unable to login app", "otp not coming", "lead not showing in crm", "app crash upload",
            "password reset not working", "portal error", "status not updating in app", "duplicate lead error",
            "add co-applicant in app", "app not working",
        ],
    },
    "legal_technical": {
        "title": "Legal & Technical",
        "desc": "Property checks: valuation report, technical/site visit, legal verification queries, title search, construction stage.",
        "aliases": [
            "valuation report pending", "technical visit not scheduled", "legal verification query",
            "valuation lower than agreement", "site visit report", "title search report", "chain documents incomplete",
            "construction stage technical", "legal opinion pending", "property revaluation",
        ],
    },
}
LABELS = list(QUEUES)
PRIORS = {"disbursal": 0.24, "documents": 0.22, "sanction": 0.18, "payout": 0.12, "tech": 0.14, "legal_technical": 0.10}

TEMPLATES: dict[str, list[str]] = {
    "disbursal": [
        "Disbursement for loan {loan_id} is pending since {days} days, please expedite",
        "When will the disbursal happen for {customer}? Sanction is done already",
        "Customer {customer} has not received the disbursal cheque from {lender}",
        "Tranche {n} release pending for {loan_id}, builder demand letter shared",
        "RTGS of {amount} to builder not done yet for file {loan_id}",
        "{lender} says disbursal is on hold, need an update",
        "DD handover date for {customer}?",
        "Loan amount not credited to seller account, file {loan_id}",
        "Disbursal request raised {days} days back but still pending with {lender} ops",
        "Part disbursement for under construction flat in {city}, what is the process",
    ],
    "documents": [
        "Customer {customer} needs to submit {doc}, what format is accepted?",
        "{lender} is asking for {doc} again, already shared twice",
        "KYC verification pending for {loan_id}",
        "PDD documents pending for {loan_id}, please share the list",
        "Which documents are required for self employed applicant in {city}?",
        "{doc} of co-applicant is missing, can we proceed with login?",
        "Bank statement of last 6 months got rejected, please advise",
        "Original property papers collection status for {loan_id}",
        "Document checklist for balance transfer case with {lender}",
        "OTC pending for {customer}, which papers are left",
    ],
    "sanction": [
        "Sanction letter not received for {loan_id}",
        "What is the status of loan approval for {customer}?",
        "{lender} rejected the file, need the rejection reason",
        "ROI in sanction letter is different from what was committed to {customer}",
        "Eligibility check for {amount} loan, salary {income} per month",
        "Can we get the sanction amount increased for {loan_id}?",
        "Login done {days} days back, no credit decision yet from {lender}",
        "Sanction validity expired, need revalidation for {loan_id}",
        "Tenure reduced in sanction, customer wants 30 years",
        "In-principle approval status for {customer} with {lender}",
    ],
    "payout": [
        "My payout for {loan_id} is not credited yet",
        "Commission for {month} disbursals still pending",
        "Payout invoice rejected due to GST number mismatch",
        "TDS deducted on my payout looks wrong",
        "What is the payout percentage for {lender} cases?",
        "When is the payout cycle for {month} disbursals?",
        "Payout statement not matching my disbursed files",
        "Please share the invoice format for raising payout",
        "Payout credited is less than the slab for {lender} files",
        "Need a statement of all payouts received in {month}",
    ],
    "tech": [
        "Unable to login to partner app, OTP not coming",
        "Lead {loan_id} not showing in CRM",
        "App crashes while uploading {doc}",
        "Password reset link is not working",
        "Portal shows error 500 when creating a new lead",
        "Status in app is not updating for {loan_id}",
        "How to add co-applicant in the app?",
        "Duplicate lead error for customer {customer}",
        "Upload of {doc} stuck at 99 percent on the portal",
        "Cannot find the loan status screen after latest update",
    ],
    "legal_technical": [
        "Property valuation report pending for {loan_id}",
        "Technical visit not scheduled for property in {city}",
        "Legal verification raised a query on {doc}",
        "Valuation came lower than agreement value, {customer} is upset",
        "Site visit done? Need the technical report for {loan_id}",
        "Title search report pending from the panel lawyer",
        "Legal team says chain documents are incomplete for {loan_id}",
        "Under construction property, technical says only {pct}% complete",
        "Panel lawyer is asking for the {doc} for title verification",
        "Property in {city} is outside the approved limits as per technical team",
    ],
}

# Held-out phrasings: only used by the `drift` split (production drift in the healing scenario)
DRIFT_TEMPLATES: dict[str, list[str]] = {
    "disbursal": [
        "Builder calling daily, bank side payment not released for {loan_id}",
        "Final tranche paisa kab release hoga? {customer} waiting",
        "Money still not transferred to seller, registry fixed on {date}",
    ],
    "documents": [
        "Credit wants fresh papers, the old {doc} is expired now",
        "Kaunse docs chahiye {customer} ke liye, salaried case hai",
        "Need to re-upload {doc} because it was blurry",
    ],
    "sanction": [
        "Credit decision kab aayega? File stuck with underwriter",
        "Customer got lower amount than applied, can credit re-look",
        "Underwriter put the file on hold for {customer}",
    ],
    "payout": [
        "Paisa nahi aaya abhi tak for my {month} cases, partner code {code}",
        "Earnings dashboard shows zero for {loan_id}",
        "Brokerage for {month} not received in my account",
    ],
    "tech": [
        "Screen goes blank after I tap submit on lead form",
        "Notifications nahi aa rahe app mein since update",
        "Cannot see my files after new app version",
    ],
    "legal_technical": [
        "Engineer ne property value kam lagayi, re-valuation possible?",
        "Advocate opinion pending, file stuck for {days} days",
        "Surveyor has not visited the flat in {city} yet",
    ],
}

CONTEXT: dict[str, list[str]] = {
    "disbursal": ["disbursal is planned next week", "after disbursal the customer wants top up"],
    "documents": ["all documents were already submitted", "KYC is complete"],
    "sanction": ["sanction letter was received last week", "file is approved by credit"],
    "payout": ["payout for this file is already fine"],
    "tech": ["I saw this in the app", "CRM shows the lead as login done"],
    "legal_technical": ["valuation is already done", "legal is cleared"],
}

LENDERS = ["Northbank HFC", "Sarvodaya Housing", "Kaveri Home Finance", "Meridian Bank", "Aravali Capital",
           "Udaan Housing", "Trident Home Loans", "Sahyadri Finance"]
CITIES = ["Pune", "Noida", "Gurugram", "Jaipur", "Indore", "Lucknow", "Bengaluru", "Thane", "Nagpur", "Surat"]
DOCS = ["ITR", "Form 16", "salary slips", "bank statement", "PAN", "Aadhaar", "sale deed", "allotment letter",
        "builder NOC", "OC certificate"]
NAMES = ["Rahul Mehta", "Priya Nair", "Amit Verma", "Sneha Kulkarni", "Vikram Rao", "Neha Gupta", "Arjun Singh",
         "Kavya Iyer", "Rohit Jain", "Ananya Das", "Sanjay Patil", "Meera Joshi"]
MONTHS = ["June", "July", "August", "September"]
OPENERS = ["", "", "Hi team, ", "Hello, ", "Urgent: ", "Dear support, ", "Team, "]
CLOSERS = ["", "", " Please check.", " Kindly revert asap.", " Customer is escalating.", " Thanks.", " Need update today."]

HINGLISH = [
    (r"\bpending\b", "pending hai"), (r"\bwhen\b", "kab"), (r"\bnot received\b", "nahi mila"),
    (r"\bplease\b", "plz"), (r"\bstill\b", "abhi tak"), (r"\burgent\b", "jaldi"), (r"\bwhat is\b", "kya hai"),
    (r"\bcustomer\b", "cust"), (r"\bnot coming\b", "nahi aa raha"), (r"\bneed\b", "chahiye"),
]
# reverse lexicon used by the `hinglish` normaliser gene
HINGLISH_REVERSE = {
    "kab": "when", "nahi": "not", "mila": "received", "aaya": "received", "aa": "", "raha": "", "rahe": "",
    "abhi": "still", "tak": "", "jaldi": "urgent", "plz": "please", "pls": "please", "kya": "what", "hai": "",
    "hoga": "will", "chahiye": "need", "kaunse": "which", "ke": "", "liye": "for", "paisa": "payment",
    "ne": "", "kam": "low", "lagayi": "", "mein": "in", "cust": "customer", "aayega": "will come",
}


HELDOUT_PER_QUEUE = 3  # last templates of every queue never appear in train -> val/test measure generalisation
LABEL_NOISE = 0.03  # annotation noise, as in any real ops dataset


@dataclass(frozen=True)
class Ticket:
    text: str
    label: str
    template: int
    ambiguous: bool
    heldout: bool = False


def _fill(tpl: str, rng: random.Random) -> str:
    return tpl.format(
        loan_id=f"HL{rng.randint(100000, 999999)}", days=rng.randint(2, 30), customer=rng.choice(NAMES),
        lender=rng.choice(LENDERS), amount=f"Rs {rng.randint(8, 95)} lakh", n=rng.randint(2, 4),
        city=rng.choice(CITIES), doc=rng.choice(DOCS), income=f"Rs {rng.randint(35, 250)}k",
        month=rng.choice(MONTHS), pct=rng.choice([30, 40, 55, 60]), code=f"P{rng.randint(1000, 9999)}",
        date=f"{rng.randint(1, 28)}/{rng.randint(1, 12)}",
    )


def _typos(text: str, p: float, rng: random.Random) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        if len(w) > 3 and not any(ch.isdigit() for ch in w) and rng.random() < p:
            i = rng.randint(1, len(w) - 2)
            op = rng.random()
            if op < 0.4:
                w = w[:i] + w[i + 1 :]
            elif op < 0.8:
                w = w[: i - 1] + w[i] + w[i - 1] + w[i + 1 :]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def _hinglish(text: str, p: float, rng: random.Random) -> str:
    for pat, rep in HINGLISH:
        if rng.random() < p:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def _make(label: str, templates: list[str], rng: random.Random, *, amb_p: float, hing_p: float,
          typo_p: float) -> Ticket:
    ti = rng.randrange(len(templates))
    body = _fill(templates[ti], rng)
    ambiguous = rng.random() < amb_p
    if ambiguous:
        other = rng.choice([q for q in LABELS if q != label])
        ctx = rng.choice(CONTEXT[other])
        body = f"{ctx[0].upper() + ctx[1:]}. {body}" if rng.random() < 0.5 else f"{body}. Also {ctx}."
    text = rng.choice(OPENERS) + body + rng.choice(CLOSERS)
    text = _hinglish(text, hing_p, rng)
    text = _typos(text, typo_p, rng)
    if rng.random() < 0.3:
        text = text.lower()
    return Ticket(text=text, label=label, template=ti, ambiguous=ambiguous,
                  heldout=ti >= len(templates) - HELDOUT_PER_QUEUE and templates is TEMPLATES.get(label))


def _noisy(t: Ticket, rng: random.Random) -> Ticket:
    if rng.random() < LABEL_NOISE:
        return Ticket(t.text, rng.choice([q for q in LABELS if q != t.label]), t.template, t.ambiguous, t.heldout)
    return t


def _labels(n: int, rng: random.Random) -> list[str]:
    return rng.choices(LABELS, weights=[PRIORS[q] for q in LABELS], k=n)


@lru_cache(maxsize=4)
def build_splits(seed: int = 7, n: int = 3000) -> dict[str, list[Ticket]]:
    rng = random.Random(seed)
    a = int(n * 0.5)
    seen = {q: TEMPLATES[q][:-HELDOUT_PER_QUEUE] for q in LABELS}
    train = [_noisy(_make(lab, seen[lab], rng, amb_p=0.3, hing_p=0.12, typo_p=0.03), rng) for lab in _labels(a, rng)]
    rest = [_noisy(_make(lab, TEMPLATES[lab], rng, amb_p=0.3, hing_p=0.12, typo_p=0.03), rng)
            for lab in _labels(n - a, rng)]
    splits = {"train": train, "val": rest[: (n - a) // 2], "test": rest[(n - a) // 2 :]}
    rr = random.Random(seed + 1)
    splits["robust"] = [
        Ticket(_typos(_hinglish(t.text, 0.7, rr), 0.12, rr), t.label, t.template, t.ambiguous) for t in splits["val"]
    ]
    rd = random.Random(seed + 2)
    splits["drift"] = [
        _make(lab, DRIFT_TEMPLATES[lab], rd, amb_p=0.25, hing_p=0.5, typo_p=0.08) for lab in _labels(900, rd)
    ]
    return splits


def export(dir_: Path, seed: int = 7) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for name, rows in build_splits(seed).items():
        with open(dir_ / f"{name}.jsonl", "w") as f:
            for t in rows:
                f.write(json.dumps(t.__dict__) + "\n")
