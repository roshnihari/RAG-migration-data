from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import pycountry
import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

PRIMARY_PAGE = "https://www.un.org/development/desa/pd/global-migration-database"
FALLBACK_PAGES = [
    "https://www.un.org/development/desa/pd/content/international-migrant-stock",
    "https://www.un.org/development/desa/pd/data/international-migration-flows",
    "https://www.un.org/development/desa/pd/content/international-migration-1",
]
LOCAL_DATASET_CANDIDATES = [
    Path("/Users/roshnihari/Downloads/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"),
]

DOWNLOAD_PATTERN = re.compile(r"\.(xlsx|xls|csv|zip)$", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")


@dataclass
class DatasetBundle:
    flows: pd.DataFrame
    map_totals: pd.DataFrame
    metadata: dict
    corpus: list[dict]


_DOWNLOAD_LOCK = Lock()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def discover_dataset_links() -> list[str]:
    session = _session()
    pages = [*FALLBACK_PAGES, PRIMARY_PAGE]
    scored_links: list[tuple[int, str]] = []

    for page in pages:
        try:
            response = session.get(page, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "").strip()
            full_url = urljoin(page, href)
            text = anchor.get_text(" ", strip=True).lower()
            path = urlparse(full_url).path.lower()
            if not (DOWNLOAD_PATTERN.search(path) or "dataset" in text or "origin" in text):
                continue

            score = 0
            if "destination and origin" in text:
                score += 100
            if "origin and destination" in text:
                score += 100
            if "destination and origin" in full_url.lower():
                score += 90
            if "origin_destination" in full_url.lower():
                score += 90
            if "origin-destination" in full_url.lower():
                score += 90
            if "by birth" in text or "by residence" in text or "by citizenship" in text:
                score += 80
            if "flow" in text or "flows" in full_url.lower():
                score += 70
            if "total, destination" in text:
                score -= 60
            if "total, origin" in text:
                score -= 30
            if "dataset" in text:
                score += 10
            if page == FALLBACK_PAGES[0]:
                score += 5

            scored_links.append((score, full_url))

    seen = set()
    ordered_links = []
    for _, link in sorted(scored_links, key=lambda item: item[0], reverse=True):
        if link not in seen:
            ordered_links.append(link)
            seen.add(link)
    return ordered_links


def download_first_dataset(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for candidate in LOCAL_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    manual_sources = sorted(cache_dir.glob("manual_un_migration_source.*"))
    if manual_sources:
        return max(manual_sources, key=lambda path: path.stat().st_mtime)

    cached_sources = sorted(cache_dir.glob("un_migration_source.*"))
    if cached_sources:
        return max(cached_sources, key=lambda path: path.stat().st_mtime)

    discovered = discover_dataset_links()
    session = _session()

    with _DOWNLOAD_LOCK:
        manual_sources = sorted(cache_dir.glob("manual_un_migration_source.*"))
        if manual_sources:
            return max(manual_sources, key=lambda path: path.stat().st_mtime)

        cached_sources = sorted(cache_dir.glob("un_migration_source.*"))
        if cached_sources:
            return max(cached_sources, key=lambda path: path.stat().st_mtime)

        for url in discovered:
            try:
                response = session.get(url, timeout=(10, 25))
                response.raise_for_status()
            except requests.RequestException:
                continue

            suffix = Path(urlparse(url).path).suffix.lower() or ".bin"
            filename = cache_dir / f"un_migration_source{suffix}"
            filename.write_bytes(response.content)
            return filename

    raise RuntimeError(
        "Unable to locate a downloadable UN migration dataset. "
        "The source page may have changed or may require a manual browser session."
    )


def _normalize_name(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "total", "world", "other", "all"}:
        return None
    return re.sub(r"\s+", " ", text)


def _to_iso3(name: str | None) -> str | None:
    if not name:
        return None

    aliases = {
        "bolivia (plurinational state of)": "Bolivia",
        "cabo verde": "Cape Verde",
        "china, hong kong sar": "Hong Kong",
        "china, macao sar": "Macao",
        "congo": "Republic of the Congo",
        "czechia": "Czech Republic",
        "democratic republic of the congo": "Congo, The Democratic Republic of the",
        "iran (islamic republic of)": "Iran",
        "lao people's democratic republic": "Laos",
        "micronesia (federated states of)": "Micronesia, Federated States of",
        "moldova (republic of)": "Moldova",
        "republic of korea": "Korea, Republic of",
        "russian federation": "Russia",
        "state of palestine": "Palestine, State of",
        "syrian arab republic": "Syria",
        "taiwan province of china": "Taiwan",
        "the former yugoslav republic of macedonia": "North Macedonia",
        "türkiye": "Turkey",
        "united republic of tanzania": "Tanzania",
        "united states of america": "United States",
        "venezuela (bolivarian republic of)": "Venezuela",
        "viet nam": "Vietnam",
    }
    candidate = aliases.get(name.lower(), name)

    try:
        return pycountry.countries.lookup(candidate).alpha_3
    except LookupError:
        return None


def _read_tables(dataset_path: Path) -> list[pd.DataFrame]:
    suffix = dataset_path.suffix.lower()
    tables: list[pd.DataFrame] = []

    if suffix == ".csv":
        tables.append(pd.read_csv(dataset_path))
        return tables

    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(dataset_path)
        for sheet_name in workbook.sheet_names:
            try:
                frame = workbook.parse(sheet_name)
            except Exception:
                continue
            if not frame.empty:
                tables.append(frame)
        return tables

    if suffix == ".zip":
        from zipfile import ZipFile

        with ZipFile(dataset_path) as archive:
            for member in archive.namelist():
                inner_suffix = Path(member).suffix.lower()
                if inner_suffix not in {".csv", ".xlsx", ".xls"}:
                    continue
                data = archive.read(member)
                if inner_suffix == ".csv":
                    tables.append(pd.read_csv(BytesIO(data)))
                else:
                    workbook = pd.ExcelFile(BytesIO(data))
                    for sheet_name in workbook.sheet_names:
                        try:
                            frame = workbook.parse(sheet_name)
                        except Exception:
                            continue
                        if not frame.empty:
                            tables.append(frame)
        return tables

    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def _pick_flow_table(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    ranked: list[tuple[int, pd.DataFrame]] = []
    for table in tables:
        normalized = [str(column).strip().lower() for column in table.columns]
        score = 0
        if any("origin" in column for column in normalized):
            score += 3
        if any("destination" in column for column in normalized):
            score += 3
        score += sum(1 for column in normalized if YEAR_PATTERN.match(column))
        ranked.append((score, table))

    if not ranked:
        raise RuntimeError("No readable tables found in the UN dataset.")

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1].copy()


def _find_stock_sheet(dataset_path: Path) -> str | None:
    if dataset_path.suffix.lower() not in {".xlsx", ".xls"}:
        return None
    workbook = pd.ExcelFile(dataset_path)
    for sheet_name in workbook.sheet_names:
        if str(sheet_name).strip().lower() == "table 1":
            return sheet_name
    return None


def _identify_dimension_column(columns: list[str], keywords: list[str]) -> str | None:
    for keyword in keywords:
        for column in columns:
            if keyword in column:
                return column
    return None


def _normalize_flows(raw: pd.DataFrame) -> pd.DataFrame:
    raw.columns = [str(column).strip() for column in raw.columns]
    lower_columns = [column.lower() for column in raw.columns]

    origin_column = _identify_dimension_column(
        lower_columns,
        ["country or area of origin", "origin", "from", "birth"],
    )
    destination_column = _identify_dimension_column(
        lower_columns,
        ["country or area of destination", "destination", "to", "residence"],
    )
    if origin_column is None or destination_column is None:
        raise RuntimeError(
            "Could not detect origin and destination columns in the downloaded UN data."
        )

    origin_actual = raw.columns[lower_columns.index(origin_column)]
    destination_actual = raw.columns[lower_columns.index(destination_column)]
    year_columns = [column for column in raw.columns if YEAR_PATTERN.match(str(column).strip())]

    if not year_columns:
        raise RuntimeError("The UN table does not contain recognizable year columns.")

    flows = raw[[origin_actual, destination_actual, *year_columns]].copy()
    flows = flows.rename(
        columns={origin_actual: "origin", destination_actual: "destination"}
    )
    flows["origin"] = flows["origin"].map(_normalize_name)
    flows["destination"] = flows["destination"].map(_normalize_name)
    flows = flows.dropna(subset=["origin", "destination"])
    flows = flows[flows["origin"] != flows["destination"]]

    long_flows = flows.melt(
        id_vars=["origin", "destination"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )
    long_flows["value"] = pd.to_numeric(long_flows["value"], errors="coerce")
    long_flows = long_flows.dropna(subset=["value"])
    long_flows = long_flows[long_flows["value"] > 0]
    long_flows["year"] = long_flows["year"].astype(int)
    long_flows["origin_iso3"] = long_flows["origin"].map(_to_iso3)
    long_flows["destination_iso3"] = long_flows["destination"].map(_to_iso3)
    long_flows = long_flows.dropna(subset=["origin_iso3", "destination_iso3"])
    return long_flows.reset_index(drop=True)


def _normalize_stock_table(dataset_path: Path) -> pd.DataFrame:
    stock = pd.read_excel(dataset_path, sheet_name="Table 1", header=10)
    stock.columns = [str(column).strip() for column in stock.columns]

    country_column = "Region, development group, country or area"
    if country_column not in stock.columns:
        raise RuntimeError("Could not detect the destination-country column in the UN stock workbook.")

    base_year_columns = []
    for column in stock.columns:
        text = str(column).strip()
        if YEAR_PATTERN.match(text):
            base_year_columns.append(column)
        elif re.fullmatch(r"(19|20)\d{2}\.0", text):
            base_year_columns.append(column)
        elif ".1" in text or ".2" in text:
            continue

    normalized_year_columns = []
    seen_years = set()
    for column in base_year_columns:
        year = int(float(str(column).strip()))
        if year not in seen_years:
            normalized_year_columns.append((column, year))
            seen_years.add(year)

    if not normalized_year_columns:
        raise RuntimeError("Could not detect year columns in the UN stock workbook.")

    keep_columns = [country_column, "Location code", *[column for column, _ in normalized_year_columns]]
    stock = stock[keep_columns].copy()
    stock = stock.rename(columns={country_column: "destination", "Location code": "location_code"})
    stock["destination"] = stock["destination"].map(_normalize_name)
    stock = stock.dropna(subset=["destination"])
    stock["destination"] = stock["destination"].str.replace(r"\*$", "", regex=True).str.strip()
    stock["destination_iso3"] = stock["destination"].map(_to_iso3)
    stock = stock.dropna(subset=["destination_iso3"])

    renamed_years = {column: str(year) for column, year in normalized_year_columns}
    stock = stock.rename(columns=renamed_years)

    long_stock = stock.melt(
        id_vars=["destination", "destination_iso3", "location_code"],
        value_vars=list(renamed_years.values()),
        var_name="year",
        value_name="value",
    )
    long_stock["value"] = pd.to_numeric(long_stock["value"], errors="coerce")
    long_stock = long_stock.dropna(subset=["value"])
    long_stock = long_stock[long_stock["value"] > 0]
    long_stock["year"] = long_stock["year"].astype(int)
    long_stock["mode"] = "destination_stock"
    return long_stock.reset_index(drop=True)


def _normalize_bilateral_workbook(dataset_path: Path) -> pd.DataFrame:
    flows = pd.read_excel(dataset_path, sheet_name="Table 1", header=10)
    flows.columns = [str(column).strip() for column in flows.columns]

    origin_column = "Region, development group, country or area of origin"
    destination_column = "Region, development group, country or area of destination"
    origin_code_column = "Location code of origin"
    destination_code_column = "Location code of destination"

    required = {origin_column, destination_column, origin_code_column, destination_code_column}
    if not required.issubset(set(flows.columns)):
        raise RuntimeError("Could not detect bilateral origin and destination columns in the UN workbook.")

    demographic_blocks = {"both sexes": [], "male": [], "female": []}
    for column in flows.columns:
        text = str(column).strip()
        if YEAR_PATTERN.match(text) or re.fullmatch(r"(19|20)\d{2}\.0", text):
            demographic_blocks["both sexes"].append((column, int(float(text))))
        elif re.fullmatch(r"(19|20)\d{2}\.1", text):
            demographic_blocks["male"].append((column, int(text.split(".")[0])))
        elif re.fullmatch(r"(19|20)\d{2}\.2", text):
            demographic_blocks["female"].append((column, int(text.split(".")[0])))

    if not demographic_blocks["both sexes"]:
        raise RuntimeError("Could not detect year columns in the bilateral UN workbook.")

    keep_columns = {
        destination_column,
        destination_code_column,
        origin_column,
        origin_code_column,
    }
    for block in demographic_blocks.values():
        for column, _ in block:
            keep_columns.add(column)

    flows = flows[list(keep_columns)].copy()
    flows = flows.rename(
        columns={
            destination_column: "destination",
            destination_code_column: "destination_code",
            origin_column: "origin",
            origin_code_column: "origin_code",
        }
    )
    flows["destination"] = flows["destination"].map(_normalize_name)
    flows["origin"] = flows["origin"].map(_normalize_name)
    flows = flows.dropna(subset=["destination", "origin"])
    flows["destination"] = flows["destination"].str.replace(r"\*$", "", regex=True).str.strip()
    flows["origin"] = flows["origin"].str.replace(r"\*$", "", regex=True).str.strip()
    flows = flows[flows["destination"] != flows["origin"]]

    long_frames = []
    id_vars = ["destination", "destination_code", "origin", "origin_code"]
    for demographic, block in demographic_blocks.items():
        if not block:
            continue
        renamed_years = {column: str(year) for column, year in block}
        demo_frame = flows[id_vars + [column for column, _ in block]].copy()
        demo_frame = demo_frame.rename(columns=renamed_years)
        demo_long = demo_frame.melt(
            id_vars=id_vars,
            value_vars=list(renamed_years.values()),
            var_name="year",
            value_name="value",
        )
        demo_long["demographic"] = demographic
        long_frames.append(demo_long)

    long_flows = pd.concat(long_frames, ignore_index=True)
    long_flows["value"] = pd.to_numeric(long_flows["value"], errors="coerce")
    long_flows = long_flows.dropna(subset=["value"])
    long_flows = long_flows[long_flows["value"] > 0]
    long_flows["year"] = long_flows["year"].astype(int)
    long_flows["destination_iso3"] = long_flows["destination"].map(_to_iso3)
    long_flows["origin_iso3"] = long_flows["origin"].map(_to_iso3)
    long_flows = long_flows.dropna(subset=["destination_iso3", "origin_iso3"])
    long_flows["mode"] = "bilateral"
    return long_flows.reset_index(drop=True)


def _build_map_totals(flows: pd.DataFrame) -> pd.DataFrame:
    if "demographic" in flows.columns:
        flows = flows[flows["demographic"] == "both sexes"].copy()

    mode = flows["mode"].iloc[0] if "mode" in flows.columns and not flows.empty else "bilateral"
    if mode == "destination_stock":
        inbound = (
            flows.groupby(["year", "destination", "destination_iso3"], as_index=False)["value"]
            .sum()
            .rename(
                columns={
                    "destination": "country",
                    "destination_iso3": "iso3",
                    "value": "inbound_total",
                }
            )
        )
        inbound["outbound_total"] = 0
        inbound["net_flow_balance"] = inbound["inbound_total"]
        return inbound.sort_values(["year", "country"]).reset_index(drop=True)

    inbound = (
        flows.groupby(["year", "destination", "destination_iso3"], as_index=False)["value"]
        .sum()
        .rename(
            columns={
                "destination": "country",
                "destination_iso3": "iso3",
                "value": "inbound_total",
            }
        )
    )
    outbound = (
        flows.groupby(["year", "origin", "origin_iso3"], as_index=False)["value"]
        .sum()
        .rename(
            columns={"origin": "country", "origin_iso3": "iso3", "value": "outbound_total"}
        )
    )

    merged = inbound.merge(outbound, how="outer", on=["year", "country", "iso3"]).fillna(0)
    merged["net_flow_balance"] = merged["inbound_total"] - merged["outbound_total"]
    return merged.sort_values(["year", "country"]).reset_index(drop=True)


def _build_corpus(flows: pd.DataFrame, metadata: dict) -> list[dict]:
    if "demographic" in flows.columns:
        flows = flows[flows["demographic"] == "both sexes"].copy()

    mode = metadata.get("mode", "bilateral")
    corpus: list[dict] = [
        {
            "id": "source-primary",
            "text": (
                "Primary source page: UN Global Migration Database. "
                "This database contains empirical statistics on international migrants "
                "by country of birth or citizenship, sex and age."
            ),
            "source": PRIMARY_PAGE,
        }
    ]

    years = sorted(flows["year"].unique().tolist())
    corpus.append(
        {
            "id": "dataset-coverage",
            "text": (
                f"The normalized {mode.replace('_', ' ')} dataset covers {len(years)} years, from "
                f"{years[0]} to {years[-1]}, with "
                f"{flows['destination'].nunique()} destination countries."
            ),
            "source": metadata.get("dataset_path", "local cache"),
        }
    )

    yearly_totals = flows.groupby("year", as_index=False)["value"].sum()
    for row in yearly_totals.to_dict("records"):
        corpus.append(
            {
                "id": f"year-{row['year']}",
                "text": (
                    f"In {row['year']}, the total bilateral migration stock represented in "
                    f"the normalized dataset is {int(row['value']):,}."
                ),
                "source": metadata.get("dataset_path", "local cache"),
            }
        )

    top_destinations = flows.groupby("destination", as_index=False)["value"].sum().sort_values("value", ascending=False).head(15)
    for row in top_destinations.to_dict("records"):
        corpus.append(
            {
                "id": f"dest-{row['destination']}",
                "text": (
                    f"{row['destination']} is one of the largest destination countries in the "
                    f"dataset with a cumulative inbound stock of {int(row['value']):,}."
                ),
                "source": metadata.get("dataset_path", "local cache"),
            }
        )

    if mode == "bilateral" and "origin" in flows.columns:
        top_corridors = (
            flows.groupby(["origin", "destination"], as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
            .head(25)
        )
        for row in top_corridors.to_dict("records"):
            corpus.append(
                {
                    "id": f"corridor-{row['origin']}-{row['destination']}",
                    "text": (
                        f"A major migration corridor in the dataset runs from {row['origin']} "
                        f"to {row['destination']} with cumulative volume {int(row['value']):,}."
                    ),
                    "source": metadata.get("dataset_path", "local cache"),
                }
            )

    return corpus


def prepare_dataset(base_dir: Path) -> DatasetBundle:
    cache_dir = base_dir / "cache"
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = download_first_dataset(cache_dir)
    mode = "bilateral"
    dataset_name = dataset_path.name.lower()
    try:
        if "destination_and_origin" in dataset_name or "destination and origin" in dataset_name:
            flows = _normalize_bilateral_workbook(dataset_path)
            mode = "bilateral"
        else:
            tables = _read_tables(dataset_path)
            raw_flow_table = _pick_flow_table(tables)
            flows = _normalize_flows(raw_flow_table)
            flows["mode"] = "bilateral"
    except RuntimeError as exc:
        stock_sheet = _find_stock_sheet(dataset_path)
        if stock_sheet is None:
            raise exc
        flows = _normalize_stock_table(dataset_path)
        mode = "destination_stock"

    map_totals = _build_map_totals(flows)

    flows_path = data_dir / "normalized_flows.csv"
    totals_path = data_dir / "map_totals.csv"
    flows.to_csv(flows_path, index=False)
    map_totals.to_csv(totals_path, index=False)

    metadata = {
        "dataset_path": str(dataset_path),
        "flows_path": str(flows_path),
        "totals_path": str(totals_path),
        "primary_source": PRIMARY_PAGE,
        "preferred_source": FALLBACK_PAGES[0],
        "fallback_sources": FALLBACK_PAGES,
        "records": int(len(flows)),
        "years": sorted(flows["year"].unique().tolist()),
        "mode": mode,
    }

    metadata_path = data_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    corpus = _build_corpus(flows, metadata)
    return DatasetBundle(flows=flows, map_totals=map_totals, metadata=metadata, corpus=corpus)


def load_cached_dataset(base_dir: Path) -> DatasetBundle | None:
    data_dir = base_dir / "data"
    flows_path = data_dir / "normalized_flows.csv"
    totals_path = data_dir / "map_totals.csv"
    metadata_path = data_dir / "metadata.json"

    if not (flows_path.exists() and totals_path.exists() and metadata_path.exists()):
        return None

    flows = pd.read_csv(flows_path)
    map_totals = pd.read_csv(totals_path)
    metadata = json.loads(metadata_path.read_text())
    corpus = _build_corpus(flows, metadata)
    return DatasetBundle(flows=flows, map_totals=map_totals, metadata=metadata, corpus=corpus)
