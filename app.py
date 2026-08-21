from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread

from countryinfo import CountryInfo
from flask import Flask, jsonify, render_template, request

from data_pipeline import load_cached_dataset, prepare_dataset
from rag_engine import MigrationRAG


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

bundle = None
rag = None
load_error = None
loading_state = "idle"
loader_thread = None
state_lock = Lock()

COUNTRY_CENTER_ALIASES = {
    "united states of america": "United States",
    "russian federation": "Russia",
    "türkiye": "Turkey",
    "viet nam": "Vietnam",
    "czechia": "Czech Republic",
}


def lookup_country_center(country: str) -> list[float] | None:
    candidate = COUNTRY_CENTER_ALIASES.get(country.lower(), country)
    try:
        info = CountryInfo(candidate).info()
    except Exception:
        return None
    latlng = info.get("latlng")
    if isinstance(latlng, list) and len(latlng) == 2:
        return [float(latlng[0]), float(latlng[1])]
    return None


def _set_loaded(new_bundle):
    global bundle, rag, load_error, loading_state
    with state_lock:
        bundle = new_bundle
        rag = MigrationRAG(new_bundle.corpus, new_bundle.flows)
        load_error = None
        loading_state = "ready"


def _load_in_background(force_refresh: bool = False):
    global bundle, rag, load_error, loading_state
    try:
        if not force_refresh:
            cached = load_cached_dataset(BASE_DIR)
            if cached is not None:
                _set_loaded(cached)
                return

        with state_lock:
            loading_state = "loading"
            load_error = None

        fresh_bundle = prepare_dataset(BASE_DIR)
        _set_loaded(fresh_bundle)
    except Exception as exc:
        with state_lock:
            load_error = str(exc)
            if bundle is None:
                loading_state = "error"
            else:
                loading_state = "ready"


def ensure_loader(force_refresh: bool = False):
    global loader_thread, loading_state
    with state_lock:
        should_start = (
            force_refresh
            or loader_thread is None
            or not loader_thread.is_alive()
        )
        if force_refresh:
            loading_state = "loading"
        elif loading_state == "idle":
            loading_state = "loading"

        if not should_start:
            return

        loader_thread = Thread(
            target=_load_in_background,
            kwargs={"force_refresh": force_refresh},
            daemon=True,
        )
        loader_thread.start()


ensure_loader()


@app.route("/")
def index():
    return render_template(
        "index.html",
        metadata=(
            bundle.metadata
            if bundle
            else {
                "primary_source": "https://www.un.org/development/desa/pd/global-migration-database",
                "preferred_source": "https://www.un.org/development/desa/pd/content/international-migrant-stock",
                "fallback_sources": [
                    "https://www.un.org/development/desa/pd/content/international-migrant-stock",
                    "https://www.un.org/development/desa/pd/data/international-migration-flows",
                    "https://www.un.org/development/desa/pd/content/international-migration-1",
                ],
                "records": 0,
            }
        ),
        years=(bundle.metadata["years"] if bundle else []),
        load_error=load_error,
        loading_state=loading_state,
    )


@app.get("/api/status")
def status():
    return jsonify(
        {
            "state": loading_state,
            "error": load_error,
            "has_data": bundle is not None,
            "years": (bundle.metadata["years"] if bundle else []),
            "records": (bundle.metadata["records"] if bundle else 0),
            "mode": (bundle.metadata.get("mode") if bundle else None),
        }
    )


@app.get("/api/map-data")
def map_data():
    if bundle is None:
        return jsonify(
            {
                "error": load_error or "Dataset is still loading from the UN source.",
                "state": loading_state,
            }
        ), 503

    year = request.args.get("year", type=int)
    metric = request.args.get("metric", default="inbound_total", type=str)
    metric = metric if metric in {"inbound_total", "outbound_total", "net_flow_balance"} else "inbound_total"

    frame = bundle.map_totals.copy()
    if year is not None:
        frame = frame[frame["year"] == year]

    if bundle.metadata.get("mode") == "bilateral" and year is not None and "origin" in bundle.flows.columns:
        top_corridors = (
            bundle.flows[
                (bundle.flows["year"] == year)
                & (
                    (bundle.flows["demographic"] == "both sexes")
                    if "demographic" in bundle.flows.columns
                    else True
                )
            ]
            .groupby(["origin", "destination"], as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
            .head(15)
            .to_dict("records")
        )
    else:
        top_corridors = []

    return jsonify(
        {
            "records": frame.to_dict("records"),
            "metric": metric,
            "year": year,
            "top_corridors": top_corridors,
            "mode": bundle.metadata.get("mode"),
        }
    )


@app.get("/api/country")
def country_detail():
    if bundle is None:
        return jsonify(
            {
                "error": load_error or "Dataset is still loading from the UN source.",
                "state": loading_state,
            }
        ), 503

    country = request.args.get("country", type=str)
    year = request.args.get("year", type=int)
    if not country:
        return jsonify({"error": "country is required"}), 400

    frame = bundle.flows.copy()
    if year is not None:
        frame = frame[frame["year"] == year]

    if bundle.metadata.get("mode") != "bilateral" or "origin" not in frame.columns:
        series = (
            frame[frame["destination"] == country][["year", "value"]]
            .sort_values("year")
            .to_dict("records")
        )
        return jsonify({"country": country, "year": year, "series": series, "mode": bundle.metadata.get("mode")})

    selected = frame[frame["destination"] == country].copy()
    demographic_breakdown = {}
    for demographic in ["both sexes", "male", "female"]:
        subset = selected[selected["demographic"] == demographic] if "demographic" in selected.columns else selected
        demographic_breakdown[demographic] = (
            subset.groupby("origin", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
            .to_dict("records")
        )

    inbound = demographic_breakdown["both sexes"]
    outbound = (
        frame[
            (frame["origin"] == country)
            & ((frame["demographic"] == "both sexes") if "demographic" in frame.columns else True)
        ]
        .groupby("destination", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .head(15)
        .to_dict("records")
    )
    return jsonify(
        {
            "country": country,
            "year": year,
            "inbound": inbound,
            "outbound": outbound,
            "demographics": demographic_breakdown,
            "center": lookup_country_center(country),
            "mode": bundle.metadata.get("mode"),
        }
    )


@app.post("/api/ask")
def ask():
    if rag is None:
        return jsonify(
            {
                "error": load_error or "RAG engine is still loading the dataset.",
                "state": loading_state,
            }
        ), 503

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    result = rag.answer(question)
    return jsonify({"answer": result.answer, "contexts": result.contexts})


@app.post("/api/reload")
def reload_data():
    ensure_loader(force_refresh=True)
    return jsonify({"state": "loading"})


if __name__ == "__main__":
    app.run(debug=True)


