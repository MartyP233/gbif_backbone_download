# GBIF Taxonomy Downloader

**Temporary project** to download the GBIF Backbone Taxonomy and prepare it for loading into Microsoft Fabric to support development / iteration for the edna project.

## Overview

This is a temporary utility project that downloads the complete GBIF Backbone Taxonomy (10+ million species) and converts it to Parquet format. The resulting `raw_gbif__backbone.parquet` file is then **manually uploaded to the Fabric Species workspace** and converted to a table for use in taxonomic classification workflows.

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
4. Save to `data/raw_gbif__backbone.parquet`

## Workflow

After running the script:

1. The Parquet file `data/raw_gbif__backbone.parquet` is generated locally
2. **Manually upload** this file to the **Fabric Species workspace**
3. In Fabric, convert the Parquet file to a table for use in taxonomic workflows
4. The table can then be used for fast taxonomic matching and classification

## Key Columns in the Backbone

The GBIF Backbone includes these useful columns:

- `taxonKey`: Unique GBIF identifier
- `scientificName`: Full scientific name
- `canonicalName`: Name without authorship
- `taxonomicStatus`: e.g., ACCEPTED, SYNONYM
- `taxonRank`: e.g., SPECIES, GENUS, FAMILY
- `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species`: Taxonomic hierarchy
- `parentKey`: Link to parent taxon

## File Sizes

- Original TSV: ~1.5 GB
- Parquet (compressed): ~400-500 MB
- Loads in seconds vs hours of API calls

## Data Source

GBIF Backbone Taxonomy: https://hosted-datasets.gbif.org/datasets/backbone/

The backbone is updated regularly. Re-run the script to get the latest version.
