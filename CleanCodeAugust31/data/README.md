# External data

The notebooks do not duplicate the curated experimental corpus. Download the
pinned working checkout into the repository's ignored `external/` directory:

```bash
python tools/fetch_data.py
```

This performs a shallow fetch of the validated upstream commit from
<https://github.com/beykyle/nucleon-nucleus-data> into
`external/nucleon-nucleus-data`. The script reports the exact upstream commit.
Pass `--revision COMMIT` only when intentionally validating a newer corpus.

Alternatively, point notebook 03 to an existing checkout whose root contains
the `data/` folder:

```powershell
$env:SCATTERING_DATA_ROOT = "C:\path\to\nucleon-nucleus-data-main"
```

The reader preserves corpus, EXFOR accession, record identity, normalization
uncertainty, and source-file provenance. It converts cross sections from b/sr
to mb/sr exactly once.
