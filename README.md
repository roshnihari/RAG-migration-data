# RAG Migration Data

This repo contains a small Flask app that:

1. Pulls raw migration data from the UN Global Migration Database workflow.
2. Normalizes bilateral migration records into a country-to-country format.
3. Builds a lightweight local RAG layer over source notes and computed summaries.
4. Serves an interactive global migration web map with year and metric filters.

## Project layout

- `app.py`: Flask server and API routes
- `data_pipeline.py`: UN data discovery, download, parsing and normalization
- `rag_engine.py`: local TF-IDF retrieval and answer generation
- `templates/index.html`: interactive Plotly world map and RAG UI
- `requirements.txt`: Python dependencies

## Quick start

```bash
cd <location of Python projects>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## How it works

The original Global Migration Database page is currently returning CloudFront `403` responses in some environments, so the loader prefers the accessible UN data pages first:

- [UN International Migrant Stock](https://www.un.org/development/desa/pd/content/international-migrant-stock)
- [UN International Migration Flows](https://www.un.org/development/desa/pd/data/international-migration-flows)
- [UN International Migration](https://www.un.org/development/desa/pd/content/international-migration-1)

The blocked page is still kept as a secondary discovery target:

- [UN Global Migration Database](https://www.un.org/development/desa/pd/global-migration-database)

The app then:

- detects likely origin and destination columns
- reshapes year columns into a long bilateral table
- maps countries to ISO-3 codes for the world choropleth
- computes inbound, outbound and net balance metrics
- builds a local retrieval corpus from source notes, yearly totals, top destinations and top corridors

## Notes

- The UN site may change workbook names or schema over time, so the parser uses heuristics instead of fixed column positions.
- If the data source cannot be downloaded at startup, the UI will show the loader error rather than crashing.
- The current map is a choropleth view of country-level migration totals. The sidebar also exposes the largest corridors for the selected year.
