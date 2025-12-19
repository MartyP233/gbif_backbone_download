# GBIF Taxonomy Downloader

Download and process the GBIF Backbone Taxonomy for fast local taxonomic classification using table joins instead of API calls.

## Overview

This tool downloads the complete GBIF Backbone Taxonomy (10+ million species) and converts it to Parquet format for efficient querying with Polars. This allows you to perform taxonomic matching locally using fast table joins instead of hitting the GBIF API.

## Quick Start

```bash
# Install dependencies
uv sync

# Download and convert GBIF taxonomy
uv run python main.py
```

This will:
1. Download the GBIF Backbone Taxonomy (~1.5 GB)
2. Extract the taxonomy data
3. Convert to Parquet format (saves ~70% space)
4. Save to `data/gbif_backbone.parquet`

## Usage

### Basic Loading

```python
import polars as pl

# Load the taxonomy (very fast!)
df = pl.read_parquet("data/gbif_backbone.parquet")

print(f"Loaded {len(df):,} taxonomic records")
```

### Exact Name Matching

```python
# Find exact match
result = df.filter(
    pl.col("scientificName") == "Homo sapiens"
)
print(result)
```

### Fuzzy Matching

```python
# Find all species in a genus
genus_species = df.filter(
    pl.col("genus") == "Panthera"
)

# Case-insensitive partial matching
df.filter(
    pl.col("scientificName").str.to_lowercase().str.contains("quercus")
)
```

### Batch Matching (Join Your Dataset)

```python
# Your dataset with species names
your_data = pl.DataFrame({
    "species_name": ["Homo sapiens", "Panthera leo", "Quercus robur"],
    "observation_count": [100, 45, 200]
})

# Join with GBIF taxonomy
matched = your_data.join(
    df,
    left_on="species_name",
    right_on="scientificName",
    how="left"
).select([
    "species_name",
    "observation_count",
    "taxonKey",
    "taxonomicStatus",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus"
])

print(matched)
```

## Key Columns

The GBIF Backbone includes these useful columns:

- `taxonKey`: Unique GBIF identifier
- `scientificName`: Full scientific name
- `canonicalName`: Name without authorship
- `taxonomicStatus`: e.g., ACCEPTED, SYNONYM
- `taxonRank`: e.g., SPECIES, GENUS, FAMILY
- `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species`: Taxonomic hierarchy
- `parentKey`: Link to parent taxon

## Advantages Over API Calls

- **Speed**: 1000x faster - no network latency
- **Reliability**: No rate limits or network issues
- **Batch Processing**: Join entire datasets at once
- **Offline**: Works without internet connection
- **Cost**: No API quota concerns

## File Sizes

- Original TSV: ~1.5 GB
- Parquet (compressed): ~400-500 MB
- Loads in seconds vs hours of API calls

## Data Source

GBIF Backbone Taxonomy: https://hosted-datasets.gbif.org/datasets/backbone/

The backbone is updated regularly. Re-run the script to get the latest version.
