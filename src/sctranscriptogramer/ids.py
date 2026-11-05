"""Map arbitrary gene identifiers to ENSEMBL peptide IDs (the ordering's id space).

The protein ordering is keyed by ENSEMBL peptide ids (e.g. ``ENSP00000252486``), but expression
matrices usually use gene symbols, ENSEMBL gene ids, or Entrez ids. :func:`map_to_protein` takes the
identifiers you have, the id *type*, and the species, and returns a ``(protein, identifier)``
dictionary DataFrame that plugs straight into
:func:`~sctranscriptogramer.pp.map_ids.project_to_ordering` as ``dictionary=``.

A gene typically has several protein isoforms; all are returned as separate rows, and
``project_to_ordering`` keeps whichever isoform is present in the ordering. The lookup requires
network access (MyGene.info or Ensembl BioMart) and can be cached to a parquet file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["map_to_protein"]

# id_type -> MyGene.info scope
_MYGENE_SCOPES = {
    "symbol": "symbol",
    "alias": "symbol,alias",
    "ensembl_gene": "ensembl.gene",
    "ensg": "ensembl.gene",
    "entrez": "entrezgene",
    "entrezgene": "entrezgene",
    "refseq": "refseq",
    "uniprot": "uniprot",
    "ensembl_protein": "ensembl.protein",
    "ensp": "ensembl.protein",
}

# species name/code -> (mygene species, biomart dataset prefix)
_SPECIES = {
    "human": ("human", "hsapiens"),
    "hs": ("human", "hsapiens"),
    "mouse": ("mouse", "mmusculus"),
    "mm": ("mouse", "mmusculus"),
    "rat": ("rat", "rnorvegicus"),
    "rn": ("rat", "rnorvegicus"),
    "yeast": ("4932", "scerevisiae"),
    "sc": ("4932", "scerevisiae"),
}

# id_type -> Ensembl BioMart filter/attribute name
_BIOMART_ATTR = {
    "symbol": "external_gene_name",
    "ensembl_gene": "ensembl_gene_id",
    "ensg": "ensembl_gene_id",
    "entrez": "entrezgene_id",
    "entrezgene": "entrezgene_id",
}


def _resolve_species(species: str) -> tuple[str, str]:
    key = species.strip().lower()
    if key not in _SPECIES:
        raise ValueError(f"unknown species {species!r}; expected one of {sorted(_SPECIES)}")
    return _SPECIES[key]


def _extract_proteins(ensembl) -> list[str]:
    """Pull ENSP ids from a MyGene ``ensembl`` field (dict or list of dicts)."""
    proteins: list[str] = []
    entries = ensembl if isinstance(ensembl, list) else [ensembl]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prot = entry.get("protein")
        if prot is None:
            continue
        proteins.extend(prot if isinstance(prot, list) else [prot])
    return proteins


def _map_mygene(ids: list[str], id_type: str, species: str) -> pd.DataFrame:
    import mygene

    scope = _MYGENE_SCOPES.get(id_type.lower())
    if scope is None:
        raise ValueError(
            f"unsupported id_type {id_type!r} for the mygene backend; "
            f"expected one of {sorted(_MYGENE_SCOPES)}"
        )
    mg_species, _ = _resolve_species(species)
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        ids,
        scopes=scope,
        fields="ensembl.protein",
        species=mg_species,
        returnall=True,
        verbose=False,
    )
    rows = []
    for hit in res["out"]:
        if hit.get("notfound"):
            continue
        ensembl = hit.get("ensembl")
        if ensembl is None:
            continue
        for protein in _extract_proteins(ensembl):
            rows.append((protein, hit["query"]))
    return pd.DataFrame(rows, columns=["protein", "identifier"])


def _map_biomart(ids: list[str], id_type: str, species: str) -> pd.DataFrame:
    from pybiomart import Dataset

    attr = _BIOMART_ATTR.get(id_type.lower())
    if attr is None:
        raise ValueError(
            f"unsupported id_type {id_type!r} for the biomart backend; "
            f"expected one of {sorted(_BIOMART_ATTR)}"
        )
    _, prefix = _resolve_species(species)
    dataset = Dataset(name=f"{prefix}_gene_ensembl", host="http://www.ensembl.org")
    df = dataset.query(
        attributes=[attr, "ensembl_peptide_id"],
        filters={attr: ids},
    )
    df.columns = ["identifier", "protein"]
    df = df.dropna(subset=["protein"])
    df = df[df["protein"] != ""]
    return df[["protein", "identifier"]]


def map_to_protein(
    ids,
    *,
    id_type: str = "symbol",
    species: str = "human",
    backend: str = "mygene",
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Map identifiers to ENSEMBL peptide ids.

    Parameters
    ----------
    ids
        Iterable of source identifiers (e.g. ``adata.var_names``).
    id_type
        Type of the input ids: ``"symbol"`` (default), ``"alias"``, ``"ensembl_gene"``/``"ensg"``,
        ``"entrez"``, ``"refseq"``, ``"uniprot"``, or ``"ensembl_protein"``/``"ensp"`` (mygene
        backend; the biomart backend supports symbol/ensembl_gene/entrez).
    species
        ``"human"``/``"mouse"``/``"rat"``/``"yeast"`` or the two-letter code (``Hs``/``Mm``/``Rn``/``Sc``).
    backend
        ``"mygene"`` (MyGene.info) or ``"biomart"`` (Ensembl BioMart).
    cache_path
        If given, read the mapping from this parquet file when present (unless ``refresh``), else
        compute and write it there.
    refresh
        Ignore any cached file and re-query.

    Returns
    -------
    pandas.DataFrame
        Columns ``protein`` (bare ENSEMBL peptide id) and ``identifier`` (the input id), one row
        per (protein, identifier) pair, deduplicated. Pass directly as
        ``project_to_ordering(dictionary=...)``.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not refresh:
            return pd.read_parquet(cache_path)

    unique_ids = [str(x) for x in pd.unique(pd.Series(list(ids), dtype="object")) if pd.notna(x)]
    if not unique_ids:
        raise ValueError("no identifiers to map")

    if backend == "mygene":
        result = _map_mygene(unique_ids, id_type, species)
    elif backend == "biomart":
        result = _map_biomart(unique_ids, id_type, species)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'mygene' or 'biomart'")

    result = (
        result.dropna()
        .astype({"protein": str, "identifier": str})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if result.empty:
        raise RuntimeError(
            "no identifiers could be mapped; check id_type/species/backend and connectivity"
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result
