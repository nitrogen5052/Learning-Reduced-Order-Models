# nucleon_nucleus_data

Curation of four corpora of nucleon-nucleus scattering data from
[EXFOR](https://nds.iaea.org/exfor/), as JSON.

| corpus | what it is |
|---|---|
| **ELM** | Elastic nucleon scattering and $(p,n)$ charge exchange to the isobaric analog state, on near-spherical targets between 10 and 200 MeV. Ported from the ELM curation notebooks. |
| **KDUQ** | A reconstruction of the experimental data used in the Koning-Delaroche global optical potential analysis. |
| **CHUQ** | A reconstruction of the experimental data used in the CH89 global optical potential analysis. |
| **Test** | Directly-measured data post-dating the KD and CH89 publications, drawn entirely from EXFOR. |

KDUQ, CHUQ and Test are specified by the tables of
`supplement_experimentalCorpora.pdf`, part of the supplemental material of 
[Pruitt, et al., *Phys. Rev. C* **107**, 014602 (2023)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.107.014602) - the paper which introduced CHUQ and KDUQ. 
The ELM corpus is that used in [Beyer et al., arxiv:2603.28599](https://arxiv.org/abs/2603.28599). 

`supplement_experimentalCorpora.pdf` lists, per corpus and sector, the EXFOR
subentry and scattering energy of every data set. Those tables are transcribed to
`spec/*.csv` and are the ground truth for what belongs in each corpus.

## What is in it

| corpus | sector | measurements | data points | EXFOR entries | coverage |
|---|---|---:|---:|---:|---:|
| elm | charge exchange | 30 | 622 | 7 | — |
| elm | elastic ay | 96 | 2757 | 36 | — |
| elm | elastic diff xs | 233 | 8061 | 74 | — |
| kduq | neutron ay | 30 | 637 | 12 | 96.8% |
| kduq | neutron elastic | 579 | 14888 | 123 | 96.2% |
| kduq | neutron total | 65 | 4497 | 35 | 95.6% |
| kduq | proton ay | 54 | 1606 | 26 | 100.0% |
| kduq | proton elastic | 115 | 4457 | 47 | 91.5% |
| kduq | proton reaction | 62 | 204 | 23 | 67.9% |
| chuq | neutron ay | 11 | 232 | 4 | 100.0% |
| chuq | neutron elastic | 57 | 1540 | 12 | 100.0% |
| chuq | proton ay | 64 | 1950 | 7 | 96.3% |
| chuq | proton elastic | 79 | 2497 | 8 | 92.9% |
| test | neutron ay | 8 | 145 | 2 | 100.0% |
| test | neutron elastic | 192 | 4335 | 19 | 99.5% |
| test | neutron total | 28 | 1393 | 8 | 100.0% |
| test | proton ay | 14 | 390 | 4 | 100.0% |
| test | proton elastic | 11 | 314 | 4 | 91.7% |
| test | proton reaction | 4 | 19 | 1 | 100.0% |

Coverage is the fraction of a sector's specification rows that produced data. It is not
defined for ELM, which is a query rather than a list of subentries.

Across the three tabulated corpora, 1419 of 1495 specification rows retrieve
successfully. Of the 76 that do not, 49 are entries x4i3 cannot parse and 10 are rows
the supplement itself marks as absent from EXFOR, leaving 17 genuine gaps. A further 46
rows retrieve but are dropped in cleaning, for want of usable uncertainties or of enough
scattering angles. All 122 are listed with reasons in `spec/known_missing.csv`.

## Layout

```
spec/                    the corpus tables, transcribed from the supplement, plus the
                         known-missing allowlist
scripts/                 PDF table extraction, EXFOR database setup, notebook scaffolding
src/nn_corpora/          the curation library
notebooks/<corpus>/      one notebook per corpus-sector
data/<corpus>/<sector>/  <Target>.json, <sector>.bib, and a per-corpus manifest.json
```

### Data

The curated data are stored in JSON, one file per corpus-sector-target. Each file is a list of
records, each a single measurement with its uncertainties and provenance. The files are
named `<Target>.json` and live in `data/<corpus>/<sector>/`. 

### Record schema

Records extend the schema `exfor_tools` writes via `Distribution.to_dataframe`, so files
stay readable by `AngularDistribution.from_dataframe` and
`EnergyDistribution.from_dataframe`. Five fields are added:

- `corpus`, `sector` — so files can be recombined after being split by target
- `projectile` — `"neutron"` or `"proton"`. The ELM corpus format keys only on the
  target, so $(n,n)$ and $(p,p)$ data for one nucleus are otherwise indistinguishable
- `notes` — every transformation applied to the record, in order
- `summed_excitation_max_MeV` — present only on the quasi-elastic records. EXFOR writes
  these as scattering, `SCT`, summed below an upper bound on the residual's excitation
  rather than as resolved elastic scattering, and this is that bound. The published
  corpora count them as elastic; the field is there so a consumer can decide whether to.
  See [Quasi-elastic data sets](#quasi-elastic-data-sets).

```json
{
    "type": "ECS_Rutherford",
    "energy": 65.0,
    "energy_err": 0.0,
    "energy_units": "MeV",
    "EXFORAccessionNumber": "O0032002",
    "source": "...",
    "x_units": "CM-degrees",
    "y_units": "no-dim",
    "data": {
        "x": [...], "y": [...], "y_err": [...],
        "systematic_normalization_error": 0.05
    },
    "corpus": "kduq", "sector": "proton_elastic",
    "projectile": "proton", "target": "Ca-40",
    "notes": ["divided by the Rutherford cross section ..."]
}
```

`y_err` is the overall per-point uncertainty; `systematic_normalization_error` is a
fractional correlated normalisation uncertainty, defaulting to 5% where EXFOR reports
none.

### Conventions

Everything is homogenised, so a consumer never has to ask what units a record is in.

| quantity | stored as | `type` |
|---|---|---|
| neutron differential elastic, $(p,n)$ | b/sr | `ECS` |
| charged-projectile differential elastic | ratio to Rutherford | `ECS_Rutherford` |
| analyzing power | dimensionless | `APower` |
| neutron total, proton reaction | b | `CS` |

Energies are in MeV and angles in CM degrees throughout.

**Charged-projectile elastic data are stored as a ratio to Rutherford whether or not
EXFOR reports them that way.** 

### Quasi-elastic data sets

Not every record in an elastic sector is elastic scattering in the strict sense. EXFOR
compiles some measurements under the scattering code `SCT`, which its dictionary defines
as "Total scattering (elastic + inelastic)", and states separately which residual
excitation the data cover. Twenty-two of the subentries the published corpora place in
elastic sectors are written this way. Ten resolve the excitation to a level and carry
the ground state, so they are elastic outright. The other twelve give only an upper
bound — an `E-LVL-MAX`, `E-EXC-MAX` or `E-EXC-MX-A` column — and are summed over the
ground state together with every level below that bound, which the experiment could not
separate. The bounds run from 30 keV on $^{93}$Nb, just under its 30.8 keV isomer, up to
800 keV.

The supplements count all twenty-two as elastic, and so does this repository: dropping
them would depart from the corpora being reproduced. But the twelve are quasi-elastic
rather than elastic, so each of their records carries `summed_excitation_max_MeV` with
its bound and a note saying so. A consumer fitting an optical potential can filter on
that field if the distinction matters for its purpose.

## Reproducing the curation
Clone the repository and its submodules:
```bash
git clone --recurse-submodules <repo-url>
cd nucleon_nucleus_data
```

Then, install the pinned versions of `exfor_tools` and `x4i3` in a virtual environment:
```bash
uv sync --extra dev                       # environment, incl. the pinned exfor_tools
source scripts/setup_exfor_db.sh 2025     # or 2024; exports X43I_DATAPATH
uv run pytest tests/ -q                   # unit and data-integrity tests
```

Finally, run the notebooks to reproduce the data:
```bash
uv run jupyter lab notebooks/             # the curation notebooks
```

## How the cleaning works

**Uncertainties.** EXFOR data sets often carry several uncertainty columns with no
machine-readable statement of how they relate, and `exfor_tools` refuses to guess. The
supplement states a rule instead — prefer `ERR-T`, then the average of `+DATA-ERR` and
`-DATA-ERR`, then `ERR-S`, `ERR-DIG`, `ERR-SYS` — and `nn_corpora.errors` implements it,
extended with `DATA-ERR` and with quadrature over numbered `DATA-ERRn` partials as the
CHUQ notes direct. It resolves about 92% of subentries automatically; the rest are
handled explicitly in `nn_corpora.overrides` or recorded as unresolved.

**Documented corrections.** The supplement's Comments describe specific corrections —
data sets a factor of 1000 out through a units mismatch, error columns to ignore, a
transcribed uncertainty of 3.0034 that should read 0.0034. These are encoded in
`nn_corpora.overrides`, each citing the note that motivates it. Target reassignments the
Comments describe need no code: the tables already list the corrected target.

**Downsampling.** Neutron total cross sections are trimmed to the tabulated energy range
and thinned to at most one datum per MeV, following the supplement: the cross section
varies slowly enough over this range that the full complement of energy bins — nearly
50,000 for one data set — carries nothing an optical model analysis can use.

**Sparse data sets.** Angular distributions with too few angles to constrain a potential
are dropped after unrolling. This is the criterion the supplement applies repeatedly when
it excludes data that "included over 100 scattering energies but only a few angles for
each energy".

## Coverage

Every row of every corpus table is accounted for: it either produces data, or it appears
in `spec/known_missing.csv` with a reason. A row that stops resolving fails its notebook
rather than disappearing quietly. The categories are:

- `absent-from-exfor` — the supplement itself marks the row as not locatable
- `x4i3-parse-failure` — the entry is in EXFOR as a `.x4` file but x4i3 cannot parse it,
  raising `BrokenNumberError` on a malformed numeric field, and so it never reaches the
  index. Affects the 2024 and 2025 databases alike
- `subentry-withdrawn` — EXFOR has renumbered or removed the subentry since the
  supplement was written
- `dropped-in-cleaning` — the data set retrieved, but did not survive cleaning: no
  usable uncertainties, too few scattering angles, or the wrong observable
- `energy-not-found`, `uncertainty-unresolved`

Where a subentry has been renumbered but the data survive elsewhere in the same entry —
the same publication and measurement campaign — retrieval substitutes it, requiring an
unambiguous energy match and recording the substitution in the coverage report and the
record's `notes`.


## Reproducing

```bash
python scripts/extract_corpus_tables.py     # PDF -> spec/*.csv
uv run pytest --nbmake notebooks/           # execute every notebook, rewriting data/
python scripts/refresh_known_missing.py     # regenerate the allowlist after review
uv run python scripts/build_manifests.py    # provenance manifests for each corpus
```

The EXFOR database version is recorded in each corpus's `manifest.json`, alongside the
`exfor_tools` and `x4i3` versions used.
