# Data included in this release

Compressed them as a zip file to upload to github

`curated/data/kduq/neutron_elastic/` and
`curated/data/test/neutron_elastic/` contain the neutron-elastic JSON records
used by Notebook 3.  They are included so a collaborator can clone the
repository and run the notebook without a separate `ScatteringData` checkout.

The files were copied from
<https://github.com/beykyle/nucleon-nucleus-data> at commit
`adc8558fe9fcf629af41bc909514f8da63d4f590`.  See
`curated/PROVENANCE.md` and `curated/UPSTREAM_README.md` before redistributing
or publishing derived datasets.

The notebook deliberately reads only these relative paths.  Its strict
selection keeps definite isotopes, elastic (not quasi-elastic) records,
positive finite cross sections with positive reported point uncertainties,
angles between 1 and 179 degrees, and energies from 5 to 200 MeV.
