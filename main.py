"""
Download GBIF Backbone Taxonomy and convert to Parquet format for efficient taxonomic matching.
"""

import zipfile
from pathlib import Path
import requests
import polars as pl
from io import BytesIO, StringIO
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
from datetime import datetime


def download_gbif_backbone(output_dir: Path = Path("data"), force_download: bool = False):
    """
    Download the GBIF Backbone Taxonomy export.
    
    The GBIF Backbone is a single synthetic management classification with
    over 10 million names used to organize the species pages on GBIF.org.
    
    Args:
        output_dir: Directory to save the downloaded and processed files
        force_download: If True, download even if file exists
    """
    output_dir.mkdir(exist_ok=True)
    
    tsv_path = output_dir / "Taxon.tsv"
    
    # Check if file already exists
    if tsv_path.exists() and not force_download:
        print(f"✓ Found existing TSV file: {tsv_path}")
        file_size = tsv_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {file_size:.1f} MB")
        print("  Skipping download. Use force_download=True to re-download.")
        return tsv_path
    
    # GBIF Backbone Taxonomy download URL
    # This is the complete Darwin Core Archive
    backbone_url = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
    
    print(f"Downloading GBIF Backbone Taxonomy from {backbone_url}")
    print("This may take several minutes as the file is ~1.5 GB...")
    
    response = requests.get(backbone_url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    # Download with progress
    zip_content = BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        zip_content.write(chunk)
        downloaded_size += len(chunk)
        if total_size > 0:
            progress = (downloaded_size / total_size) * 100
            print(f"\rDownloading: {progress:.1f}% ({downloaded_size / 1024 / 1024:.1f} MB)", end="")
    
    print("\n✓ Download complete!")
    
    # Extract all contents of the zip file
    print("Extracting all files from archive...")
    zip_content.seek(0)
    with zipfile.ZipFile(zip_content) as z:
        # List all files in the archive
        print(f"Found {len(z.namelist())} files in archive:")
        for filename in z.namelist():
            print(f"  - {filename}")
        
        # Extract all files
        z.extractall(output_dir)
        print(f"✓ Extracted all files to {output_dir}")
    
    return output_dir / "Taxon.tsv"


def convert_tsv_to_parquet(tsv_path: Path, output_path: Path = None, show_schema: bool = False):
    """
    Convert a single TSV file to Parquet format using Polars.
    
    Uses Python's csv module to properly handle embedded newlines in quoted fields,
    then loads into Polars for efficient conversion to parquet.
    
    Args:
        tsv_path: Path to the TSV file
        output_path: Path for the output parquet file (default: same name with .parquet extension)
        show_schema: If True, display schema and columns
    """
    if output_path is None:
        output_path = tsv_path.with_suffix('.parquet')
    
    print(f"Reading {tsv_path.name}...")
    
    # Increase field size limit for large fields in GBIF data
    import sys
    csv.field_size_limit(sys.maxsize)
    
    # Read and clean using Python's csv module (handles complex quoting properly)
    rows = []
    with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Get header
        header = next(reader)
        
        # Read all rows with progress indication
        for i, row in enumerate(reader):
            # Clean embedded newlines from fields
            cleaned_row = [field.replace('\n', ' ').replace('\r', ' ') if field else None for field in row]
            rows.append(cleaned_row)
            
            if (i + 1) % 500000 == 0:
                print(f"  Processed {i + 1:,} rows...")
    
    print(f"  Loaded {len(rows):,} rows")
    
    # Convert to Polars DataFrame
    df = pl.DataFrame({col: [row[i] if i < len(row) else None for row in rows] for i, col in enumerate(header)})
    
    if show_schema:
        print(f"  Dataset shape: {df.shape}")
        print(f"  Columns: {df.columns}")
        print(f"  Schema: {df.schema}")
    
    # Save to parquet with compression
    df.write_parquet(
        output_path,
        compression="zstd",
        compression_level=3,
    )
    
    # Show file size comparison
    tsv_size = tsv_path.stat().st_size / (1024 * 1024)
    parquet_size = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = (1 - parquet_size / tsv_size) * 100
    
    print(f"  ✓ Saved to {output_path.name} ({parquet_size:.1f} MB, {compression_ratio:.1f}% smaller)\n")
    
    return df, output_path


def convert_all_tsvs_to_parquet(data_dir: Path = Path("data")):
    """
    Convert all TSV files in the data directory to Parquet format.
    
    Args:
        data_dir: Directory containing the TSV files
        
    Returns:
        Dictionary mapping TSV names to their DataFrames and Parquet paths
    """
    tsv_files = sorted(data_dir.glob("*.tsv"))
    
    if not tsv_files:
        print(f"No TSV files found in {data_dir}")
        return {}
    
    print(f"Found {len(tsv_files)} TSV files to convert:\n")
    for tsv_file in tsv_files:
        print(f"  - {tsv_file.name}")
    print()
    
    results = {}
    total_tsv_size = 0
    total_parquet_size = 0
    
    for i, tsv_file in enumerate(tsv_files, 1):
        print(f"[{i}/{len(tsv_files)}] Converting {tsv_file.name}...")
        
        # Show detailed schema only for the main Taxon file
        show_schema = (tsv_file.name == "Taxon.tsv")
        
        df, parquet_path = convert_tsv_to_parquet(tsv_file, show_schema=show_schema)
        results[tsv_file.name] = {"dataframe": df, "parquet_path": parquet_path}
        
        total_tsv_size += tsv_file.stat().st_size
        total_parquet_size += parquet_path.stat().st_size
    
    # Summary
    print("="*60)
    print("Conversion Summary:")
    print(f"  Converted {len(tsv_files)} files")
    print(f"  Total TSV size:     {total_tsv_size / (1024 * 1024):.1f} MB")
    print(f"  Total Parquet size: {total_parquet_size / (1024 * 1024):.1f} MB")
    print(f"  Total space saved:  {(1 - total_parquet_size / total_tsv_size) * 100:.1f}%")
    print("="*60)
    
    return results


def parse_meta_xml(meta_path: Path) -> Dict:
    """
    Parse the meta.xml file to extract schema information.
    
    Args:
        meta_path: Path to meta.xml file
        
    Returns:
        Dictionary with core and extension information
    """
    tree = ET.parse(meta_path)
    root = tree.getroot()
    
    # Define namespace
    ns = {'dwc': 'http://rs.tdwg.org/dwc/text/'}
    
    schema = {
        'core': None,
        'extensions': []
    }
    
    # Parse core table
    core = root.find('dwc:core', ns)
    if core is not None:
        core_info = {
            'type': 'core',
            'rowType': core.get('rowType', '').split('/')[-1],
            'file': core.find('dwc:files/dwc:location', ns).text,
            'fields': []
        }
        
        # Get ID field
        id_elem = core.find('dwc:id', ns)
        if id_elem is not None:
            core_info['id_column'] = int(id_elem.get('index', 0))
        
        # Get all fields
        for field in core.findall('dwc:field', ns):
            term = field.get('term', '').split('/')[-1]
            index = int(field.get('index', -1))
            core_info['fields'].append({'index': index, 'name': term})
        
        schema['core'] = core_info
    
    # Parse extensions
    for extension in root.findall('dwc:extension', ns):
        ext_info = {
            'type': 'extension',
            'rowType': extension.get('rowType', '').split('/')[-1],
            'file': extension.find('dwc:files/dwc:location', ns).text,
            'fields': []
        }
        
        # Get coreid field
        coreid_elem = extension.find('dwc:coreid', ns)
        if coreid_elem is not None:
            ext_info['coreid_column'] = int(coreid_elem.get('index', 0))
        
        # Get all fields
        for field in extension.findall('dwc:field', ns):
            term = field.get('term', '').split('/')[-1]
            index = int(field.get('index', -1))
            ext_info['fields'].append({'index': index, 'name': term})
        
        schema['extensions'].append(ext_info)
    
    return schema


def parse_eml_xml(eml_path: Path) -> Dict:
    """
    Parse the eml.xml file to extract dataset metadata.
    
    Args:
        eml_path: Path to eml.xml file
        
    Returns:
        Dictionary with dataset metadata
    """
    tree = ET.parse(eml_path)
    root = tree.getroot()
    
    # Define namespaces
    ns = {
        'eml': 'eml://ecoinformatics.org/eml-2.1.1',
        'dc': 'http://purl.org/dc/terms/'
    }
    
    metadata = {}
    
    # Get title
    title_elem = root.find('.//eml:dataset/eml:title', ns)
    if title_elem is not None:
        metadata['title'] = title_elem.text
    
    # Get abstract
    abstract_paras = root.findall('.//eml:dataset/eml:abstract/eml:para', ns)
    if abstract_paras:
        metadata['abstract'] = '\n\n'.join(p.text or '' for p in abstract_paras)
    
    # Get license
    license_elem = root.find('.//eml:dataset/eml:intellectualRights/eml:para/eml:ulink/eml:citetitle', ns)
    if license_elem is not None:
        metadata['license'] = license_elem.text
    
    # Get citation
    citation_elem = root.find('.//eml:additionalMetadata/eml:metadata/gbif/citation', {'eml': 'eml://ecoinformatics.org/eml-2.1.1', 'gbif': ''})
    if citation_elem is not None:
        metadata['citation'] = citation_elem.text
    
    # Get date stamp
    date_elem = root.find('.//eml:additionalMetadata/eml:metadata/gbif/dateStamp', {'eml': 'eml://ecoinformatics.org/eml-2.1.1', 'gbif': ''})
    if date_elem is not None:
        metadata['dateStamp'] = date_elem.text
    
    return metadata


def generate_data_dictionary(schema: Dict, data_dir: Path) -> str:
    """
    Generate a human-readable data dictionary from the schema.
    
    Args:
        schema: Parsed schema from meta.xml
        data_dir: Directory containing the data files
        
    Returns:
        Markdown formatted data dictionary
    """
    lines = ["# GBIF Backbone Taxonomy - Data Dictionary\n"]
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Core table
    if schema['core']:
        core = schema['core']
        lines.append(f"## Core Table: {core['file']}")
        lines.append(f"\n**Type:** {core['rowType']}")
        lines.append(f"\n**Primary Key:** Column {core.get('id_column', 0)} (taxonID)\n")
        
        # Get row count if parquet exists
        parquet_path = data_dir / Path(core['file']).with_suffix('.parquet')
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            lines.append(f"**Row Count:** {len(df):,}\n")
        
        lines.append("\n### Columns\n")
        lines.append("| Index | Column Name | Description |")
        lines.append("|-------|-------------|-------------|")
        
        # Add ID column
        lines.append(f"| {core.get('id_column', 0)} | taxonID | Primary identifier (ID column) |")
        
        # Add all fields
        for field in sorted(core['fields'], key=lambda x: x['index']):
            lines.append(f"| {field['index']} | {field['name']} | Darwin Core term |")
        
        lines.append("")
    
    # Extension tables
    if schema['extensions']:
        lines.append("## Extension Tables\n")
        lines.append("Extension tables contain additional information linked to the core Taxon table via the taxonID.\n")
        
        for ext in schema['extensions']:
            lines.append(f"### {ext['file']}")
            lines.append(f"\n**Type:** {ext['rowType']}")
            lines.append(f"\n**Foreign Key:** Column {ext.get('coreid_column', 0)} references Taxon.taxonID\n")
            
            # Get row count if parquet exists
            parquet_path = data_dir / Path(ext['file']).with_suffix('.parquet')
            if parquet_path.exists():
                df = pl.read_parquet(parquet_path)
                lines.append(f"**Row Count:** {len(df):,}\n")
            
            lines.append("\n#### Columns\n")
            lines.append("| Index | Column Name | Description |")
            lines.append("|-------|-------------|-------------|")
            
            # Add coreid column
            lines.append(f"| {ext.get('coreid_column', 0)} | taxonID | Foreign key to core table |")
            
            # Add all fields
            for field in sorted(ext['fields'], key=lambda x: x['index']):
                lines.append(f"| {field['index']} | {field['name']} | Darwin Core / Dublin Core term |")
            
            lines.append("")
    
    return "\n".join(lines)


def generate_readme(metadata: Dict, schema: Dict, results: Dict, data_dir: Path) -> str:
    """
    Generate a comprehensive README for the dataset.
    
    Args:
        metadata: Parsed metadata from eml.xml
        schema: Parsed schema from meta.xml
        results: Results from parquet conversion
        data_dir: Directory containing the data files
        
    Returns:
        Markdown formatted README
    """
    lines = [f"# {metadata.get('title', 'GBIF Backbone Taxonomy')}\n"]
    
    # Overview
    lines.append("## Overview\n")
    if 'abstract' in metadata:
        # Take first paragraph only for README
        first_para = metadata['abstract'].split('\n\n')[0]
        lines.append(first_para + "\n")
    
    # Dataset Information
    lines.append("## Dataset Information\n")
    if 'license' in metadata:
        lines.append(f"**License:** {metadata['license']}\n")
    if 'dateStamp' in metadata:
        lines.append(f"**Last Updated:** {metadata['dateStamp']}\n")
    if 'citation' in metadata:
        lines.append(f"**Citation:** {metadata['citation']}\n")
    
    # Files
    lines.append("## Data Files\n")
    lines.append("This dataset consists of one core table and several extension tables:\n")
    
    # Core table
    if schema['core']:
        core = schema['core']
        lines.append(f"### Core Table: `{core['file']}`")
        parquet_file = Path(core['file']).with_suffix('.parquet')
        if core['file'] in results:
            row_count = len(results[core['file']]['dataframe'])
            parquet_size = results[core['file']]['parquet_path'].stat().st_size / (1024 * 1024)
            lines.append(f"- **Parquet file:** `{parquet_file.name}`")
            lines.append(f"- **Records:** {row_count:,}")
            lines.append(f"- **Size:** {parquet_size:.1f} MB")
            lines.append(f"- **Columns:** {len(core['fields']) + 1}")
        lines.append("")
    
    # Extensions
    if schema['extensions']:
        lines.append("### Extension Tables\n")
        for ext in schema['extensions']:
            parquet_file = Path(ext['file']).with_suffix('.parquet')
            lines.append(f"#### `{ext['file']}`")
            if ext['file'] in results:
                row_count = len(results[ext['file']]['dataframe'])
                parquet_size = results[ext['file']]['parquet_path'].stat().st_size / (1024 * 1024)
                lines.append(f"- **Parquet file:** `{parquet_file.name}`")
                lines.append(f"- **Records:** {row_count:,}")
                lines.append(f"- **Size:** {parquet_size:.1f} MB")
                lines.append(f"- **Columns:** {len(ext['fields']) + 1}")
                lines.append(f"- **Relationship:** Links to core table via taxonID")
            lines.append("")
    
    # Usage examples
    lines.append("## Usage Examples\n")
    lines.append("### Loading Data with Polars\n")
    lines.append("```python")
    lines.append("import polars as pl")
    lines.append("")
    lines.append("# Load the core taxonomy table")
    lines.append('taxon = pl.read_parquet("data/Taxon.parquet")')
    lines.append("")
    lines.append("# Load vernacular names")
    lines.append('vernacular = pl.read_parquet("data/VernacularName.parquet")')
    lines.append("")
    lines.append("# Join to get common names for species")
    lines.append("species_with_names = taxon.join(")
    lines.append("    vernacular,")
    lines.append("    left_on='id',")
    lines.append("    right_on='id',")
    lines.append("    how='left'")
    lines.append(")")
    lines.append("```\n")
    
    lines.append("### Querying Taxonomy\n")
    lines.append("```python")
    lines.append("# Find all species in a genus")
    lines.append('genus_species = taxon.filter(pl.col("genus") == "Quercus")')
    lines.append("")
    lines.append("# Get full classification for a taxon")
    lines.append('oak = taxon.filter(pl.col("canonicalName") == "Quercus robur").select([')
    lines.append('    "kingdom", "phylum", "class", "order", "family", "genus", "scientificName"')
    lines.append("])")
    lines.append("```\n")
    
    # Data Dictionary Reference
    lines.append("## Data Dictionary\n")
    lines.append("For detailed information about all columns and their meanings, see [DATA_DICTIONARY.md](DATA_DICTIONARY.md).\n")
    
    # Schema diagram
    lines.append("## Data Model\n")
    lines.append("```")
    lines.append("┌─────────────────┐")
    lines.append("│   Taxon (Core)  │")
    lines.append("│   Primary Key:  │")
    lines.append("│     taxonID     │")
    lines.append("└────────┬────────┘")
    lines.append("         │")
    lines.append("         │ Referenced by (taxonID)")
    lines.append("         │")
    lines.append("    ┌────┴────┬─────────────┬──────────────┬─────────────┬──────────────┐")
    lines.append("    │         │             │              │             │              │")
    lines.append("┌───▼───┐ ┌──▼──┐ ┌─────────▼───┐ ┌───────▼──────┐ ┌───▼────┐ ┌──────▼─────┐")
    lines.append("│ Desc- │ │Dist-│ │ Vernacular  │ │  Multimedia  │ │ Refer- │ │  Types &   │")
    lines.append("│ription│ │ribu-│ │    Name     │ │              │ │ ence   │ │  Specimen  │")
    lines.append("│       │ │tion │ │             │ │              │ │        │ │            │")
    lines.append("└───────┘ └─────┘ └─────────────┘ └──────────────┘ └────────┘ └────────────┘")
    lines.append("```\n")
    
    # Additional resources
    lines.append("## Additional Resources\n")
    lines.append("- **GBIF Website:** https://www.gbif.org")
    lines.append("- **Darwin Core Standard:** https://dwc.tdwg.org/")
    lines.append("- **Download Archive:** https://hosted-datasets.gbif.org/datasets/backbone/\n")
    
    return "\n".join(lines)


def process_metadata(data_dir: Path = Path("data")):
    """
    Process XML metadata files and generate documentation.
    
    Args:
        data_dir: Directory containing the data and XML files
    """
    meta_path = data_dir / "meta.xml"
    eml_path = data_dir / "eml.xml"
    
    if not meta_path.exists() or not eml_path.exists():
        print("⚠ XML metadata files not found. Skipping documentation generation.")
        return
    
    print("\nProcessing metadata files...")
    print("="*60 + "\n")
    
    # Parse XML files
    print("Parsing meta.xml...")
    schema = parse_meta_xml(meta_path)
    
    print("Parsing eml.xml...")
    metadata = parse_eml_xml(eml_path)
    
    print("✓ Metadata parsed successfully\n")
    
    return schema, metadata


def main():
    """
    Main workflow: Download GBIF taxonomy, convert to Parquet, and generate documentation.
    """
    print("GBIF Taxonomy Downloader and Converter")
    print("="*60 + "\n")
    
    # Step 1: Download the GBIF backbone (will skip if already exists)
    data_dir = Path("data")
    tsv_path = download_gbif_backbone(output_dir=data_dir)
    
    # Step 2: Convert all TSV files to Parquet
    print("\nConverting all TSV files to Parquet format...")
    print("="*60 + "\n")
    results = convert_all_tsvs_to_parquet(data_dir)
    
    # Step 3: Process metadata and generate documentation
    metadata_result = process_metadata(data_dir)
    
    if metadata_result:
        schema, metadata = metadata_result
        
        print("Generating documentation...")
        
        # Generate data dictionary
        data_dict = generate_data_dictionary(schema, data_dir)
        dict_path = data_dir / "DATA_DICTIONARY.md"
        with open(dict_path, 'w', encoding='utf-8') as f:
            f.write(data_dict)
        print(f"✓ Data dictionary saved to {dict_path}")
        
        # Generate README
        readme = generate_readme(metadata, schema, results, data_dir)
        readme_path = data_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)
        print(f"✓ README saved to {readme_path}")
    
    print("\n" + "="*60)
    print("Done! You can now use the parquet files for fast data access.")
    print(f"\nParquet files saved in: {data_dir}")
    print("\nAvailable files:")
    for tsv_name, result in results.items():
        parquet_name = result['parquet_path'].name
        row_count = len(result['dataframe'])
        print(f"  - {parquet_name} ({row_count:,} rows)")
    
    if metadata_result:
        print(f"\nDocumentation:")
        print(f"  - {data_dir}/README.md")
        print(f"  - {data_dir}/DATA_DICTIONARY.md")
    
    print("="*60)


if __name__ == "__main__":
    main()
