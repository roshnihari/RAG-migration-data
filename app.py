from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from data_pipeline import prepare_dataset
from rag_engine import MigrationRAG


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

bundle = None
rag = None
load_error = None

try:
    bundle = prepare_dataset(BASE_DIR)
    rag = MigrationRAG(bundle.corpus, bundle.flows)
except Exception as exc:
    load_error = str(exc)


@app.route("/")
def index():
    return render_template(
        "index.html",
        metadata=(bundle.metadata if bundle else {"primary_source": "https://www.un.org/development/desa/pd/global-migration-database", "records": 0}),
        years=(bundle.metadata["years"] if bundle else []),
        load_error=load_error,
    )


@app.get("/api/map-data")
def map_data():
    if bundle is None:
        return jsonify({"error": load_error or "Dataset could not be loaded."}), 503

    year = request.args.get("year", type=int)
    metric = request.args.get("metric", default="inbound_total", type=str)
    metric = metric if metric in {"inbound_total", "outbound_total", "net_flow_balance"} else "inbound_total"

    frame = bundle.map_totals.copy()
    if year is not None:
        frame = frame[frame["year"] == year]

    top_corridors = (
        bundle.flows[bundle.flows["year"] == year]
        .groupby(["origin", "destination"], as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .head(15)
        .to_dict("records")
        if year is not None
        else []
    )

    return jsonify(
        {
            "records": frame.to_dict("records"),
            "metric": metric,
            "year": year,
            "top_corridors": top_corridors,
        }
    )


@app.get("/api/country")
def country_detail():
    if bundle is None:
        return jsonify({"error": load_error or "Dataset could not be loaded."}), 503

    country = request.args.get("country", type=str)
    year = request.args.get("year", type=int)
    if not country:
        return jsonify({"error": "country is required"}), 400

    frame = bundle.flows.copy()
    if year is not None:
        frame = frame[frame["year"] == year]

    inbound = (
        frame[frame["destination"] == country]
        .groupby("origin", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .head(10)
        .to_dict("records")
    )
    outbound = (
        frame[frame["origin"] == country]
        .groupby("destination", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .head(10)
        .to_dict("records")
    )
    return jsonify({"country": country, "year": year, "inbound": inbound, "outbound": outbound})


@app.post("/api/ask")
def ask():
    if rag is None:
        return jsonify({"error": load_error or "RAG engine is unavailable."}), 503

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    result = rag.answer(question)
    return jsonify({"answer": result.answer, "contexts": result.contexts})


if __name__ == "__main__":
    app.run(debug=True)
