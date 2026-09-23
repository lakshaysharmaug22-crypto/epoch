"""HomeLoanQA: a deterministic, fictional home-loan policy corpus with extractive QA pairs.

Every fact appears once, inside long documents full of near-duplicate distractor facts from other lenders,
so retrieval quality, chunking and reranking genuinely matter. Swap in your own corpus with
`EPOCH_RAG_CORPUS=path.jsonl` (fields: id, title, text) and `EPOCH_RAG_QA=path.jsonl` (question, answer).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache

LENDERS = ["Northbank HFC", "Sarvodaya Housing", "Kaveri Home Finance", "Meridian Bank", "Aravali Capital",
           "Udaan Housing", "Trident Home Loans", "Sahyadri Finance", "Konark Bank", "Pinnacle HFC"]

ATTRS = {
    "max_tenure": (lambda r: f"{r.choice([20, 25, 30, 35])} years", "the maximum loan tenure",
                   ["What is the maximum loan tenure at {L}?", "For how many years can a {L} home loan run at most?"]),
    "max_ltv": (lambda r: f"{r.choice([70, 75, 80, 85, 90])} percent", "the maximum loan-to-value ratio",
                ["What is the maximum LTV offered by {L}?", "Up to what loan-to-value ratio does {L} lend?"]),
    "roi_floor": (lambda r: f"{r.randint(835, 990) / 100:.2f} percent", "the starting interest rate",
                  ["What is the starting interest rate at {L}?", "From what rate of interest do {L} home loans start?"]),
    "processing_fee": (lambda r: r.choice(["0.25 percent of the loan amount", "0.5 percent of the loan amount",
                                           "1 percent of the loan amount", f"a flat Rs {r.randint(5, 15)},000"]),
                       "the processing fee",
                       ["What processing fee does {L} charge?", "How much is the processing fee at {L}?"]),
    "min_cibil": (lambda r: str(r.randint(650, 760)), "the minimum credit score required",
                  ["What minimum CIBIL score does {L} require?", "Which credit score is the minimum at {L}?"]),
    "foreclosure": (lambda r: r.choice(["nil for floating rate loans", "2 percent of the outstanding principal",
                                        "3 percent of the outstanding principal"]), "the foreclosure charge",
                    ["What is the foreclosure charge at {L}?", "How much does {L} charge for closing the loan early?"]),
    "turnaround": (lambda r: f"{r.randint(3, 12)} working days", "the sanction turnaround time",
                   ["How long does {L} take to sanction a loan?", "What is the sanction turnaround time at {L}?"]),
    "max_age": (lambda r: f"{r.randint(60, 70)} years at loan maturity", "the maximum borrower age",
                ["What is the maximum borrower age allowed by {L}?", "Till what age can a borrower repay a {L} loan?"]),
    "min_income": (lambda r: f"Rs {r.randint(15, 50)},000 per month", "the minimum net income",
                   ["What minimum monthly income does {L} need?", "What is the minimum net income for a {L} home loan?"]),
    "topup": (lambda r: f"up to Rs {r.randint(10, 60)} lakh", "the top-up loan limit",
              ["How much top-up loan can {L} give?", "What is the top-up limit on a {L} balance transfer?"]),
}
DOC_GROUPS = {
    "Product overview": ["max_tenure", "max_ltv", "roi_floor", "processing_fee"],
    "Eligibility norms": ["min_cibil", "max_age", "min_income"],
    "Servicing and charges": ["foreclosure", "turnaround", "topup"],
}
FILLER = [
    "Applications are accepted through partner channels across India and are processed digitally.",
    "Rates are linked to an external benchmark and may be revised at reset dates.",
    "Salaried and self-employed applicants are both eligible subject to credit assessment.",
    "The lender may ask for additional documents depending on the profile of the applicant.",
    "Co-applicants are encouraged where the primary applicant's income is variable.",
    "All charges are exclusive of applicable taxes.",
    "Property must have clear and marketable title as per the panel advocate's opinion.",
    "Customers can track their application on the partner app after login.",
]


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QA:
    question: str
    answer: str
    lender: str
    attr: str
    fact: str


def _perturb(q: str, r: random.Random) -> str:
    q = q.replace("maximum", "max").replace("minimum", "min") if r.random() < 0.6 else q
    words = q.split()
    for i, w in enumerate(words):
        if len(w) > 4 and r.random() < 0.12:
            j = r.randint(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1 :]
    q = " ".join(words)
    return q.lower() if r.random() < 0.5 else q


@lru_cache(maxsize=2)
def build(seed: int = 11) -> tuple[list[Doc], dict[str, list[QA]]]:
    if os.environ.get("EPOCH_RAG_CORPUS") and os.environ.get("EPOCH_RAG_QA"):
        return _load_custom()
    r = random.Random(seed)
    docs: list[Doc] = []
    qas: list[QA] = []
    for li, lender in enumerate(LENDERS):
        for title, attrs in DOC_GROUPS.items():
            sents = []
            for a in attrs:
                gen, phrase, qtpls = ATTRS[a]
                value = gen(r)
                fact = f"At {lender}, {phrase} is {value}."
                sents.append(fact)
                for t in qtpls:
                    qas.append(QA(t.format(L=lender), value, lender, a, fact))
            # distractors: comparisons that mention other lenders' generic terms without values
            other = LENDERS[(li + 3) % len(LENDERS)]
            sents.append(f"Unlike {other}, {lender} publishes its schedule of charges on its website.")
            sents += r.sample(FILLER, 4)
            r.shuffle(sents)
            docs.append(Doc(f"{li:02d}-{title[:3].lower()}", f"{lender} — {title}", " ".join(sents)))
    r.shuffle(qas)
    n = len(qas)
    val, test = qas[: int(n * 0.6)], qas[int(n * 0.6) :]
    rr = random.Random(seed + 1)
    robust = [QA(_perturb(q.question, rr), q.answer, q.lender, q.attr, q.fact) for q in val]
    return docs, {"val": val, "test": test, "robust": robust, "train": []}


def _load_custom() -> tuple[list[Doc], dict[str, list[QA]]]:
    with open(os.environ["EPOCH_RAG_CORPUS"]) as f:
        docs = [Doc(**json.loads(line)) for line in f]
    with open(os.environ["EPOCH_RAG_QA"]) as f:
        rows = [json.loads(line) for line in f]
    qas = [QA(r["question"], r["answer"], r.get("lender", ""), r.get("attr", ""), r.get("fact", r["answer"])) for r in rows]
    n = len(qas)
    return docs, {"val": qas[: int(n * 0.6)], "test": qas[int(n * 0.6) :], "robust": [], "train": []}
