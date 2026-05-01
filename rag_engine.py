from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    answer: str
    contexts: list[dict]


class MigrationRAG:
    def __init__(self, corpus: list[dict], flows: pd.DataFrame):
        self.corpus = corpus
        self.flows = flows
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([item["text"] for item in corpus])

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]
        results = []
        for index in ranked_indexes:
            item = dict(self.corpus[index])
            item["score"] = float(scores[index])
            results.append(item)
        return results

    def answer(self, question: str) -> RetrievalResult:
        contexts = self.retrieve(question)
        answer = self._build_answer(question, contexts)
        return RetrievalResult(answer=answer, contexts=contexts)

    def _build_answer(self, question: str, contexts: list[dict]) -> str:
        year = self._extract_year(question)
        country = self._match_country(question)

        grounded_lines = []
        if year is not None:
            yearly = self.flows[self.flows["year"] == year]
            if not yearly.empty:
                total = int(yearly["value"].sum())
                grounded_lines.append(
                    f"For {year}, the normalized dataset totals {total:,} across all retained bilateral records."
                )

                top_corridors = (
                    yearly.groupby(["origin", "destination"], as_index=False)["value"]
                    .sum()
                    .sort_values("value", ascending=False)
                    .head(3)
                )
                if not top_corridors.empty:
                    summary = "; ".join(
                        f"{row.origin} -> {row.destination} ({int(row.value):,})"
                        for row in top_corridors.itertuples(index=False)
                    )
                    grounded_lines.append(f"Top corridors in {year}: {summary}.")

        if country:
            related = self.flows[
                (self.flows["origin"].str.lower() == country.lower())
                | (self.flows["destination"].str.lower() == country.lower())
            ]
            if not related.empty:
                inbound = (
                    related[related["destination"].str.lower() == country.lower()]
                    .groupby("origin", as_index=False)["value"]
                    .sum()
                    .sort_values("value", ascending=False)
                    .head(3)
                )
                outbound = (
                    related[related["origin"].str.lower() == country.lower()]
                    .groupby("destination", as_index=False)["value"]
                    .sum()
                    .sort_values("value", ascending=False)
                    .head(3)
                )
                if not inbound.empty:
                    grounded_lines.append(
                        "Largest inbound links: "
                        + "; ".join(
                            f"{row.origin} ({int(row.value):,})" for row in inbound.itertuples(index=False)
                        )
                        + "."
                    )
                if not outbound.empty:
                    grounded_lines.append(
                        "Largest outbound links: "
                        + "; ".join(
                            f"{row.destination} ({int(row.value):,})"
                            for row in outbound.itertuples(index=False)
                        )
                        + "."
                    )

        retrieved = " ".join(context["text"] for context in contexts[:3])
        grounded_lines.append(f"Retrieved context: {retrieved}")
        return " ".join(grounded_lines)

    def _extract_year(self, text: str) -> int | None:
        match = re.search(r"\b(19|20)\d{2}\b", text)
        return int(match.group(0)) if match else None

    def _match_country(self, text: str) -> str | None:
        lowered = text.lower()
        countries = set(self.flows["origin"].dropna().tolist()) | set(
            self.flows["destination"].dropna().tolist()
        )
        for country in sorted(countries, key=len, reverse=True):
            if country.lower() in lowered:
                return country
        return None
