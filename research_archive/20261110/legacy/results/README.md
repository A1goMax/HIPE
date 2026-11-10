# Historical PE result evidence

historical-evidence.json contains selected numerical records from the legacy PE experiment families. The table numbers identify related topics in the manuscript. Protocol equivalence and assignment to individual manuscript cells remain unresolved.

| Related table | Historical evidence | Records |
| --- | --- | ---: |
| 1 | Synthetic task history endpoints and declared configuration | 109 |
| 2 | WikiText 20M and 60M last logged validation | 260 |
| 3 | C4 300M last logged validation for identified non-supplementary methods | 18 |
| 4 | SST-2 and CoLA saved downstream best metrics and configuration | 171 |
| 5 | 60M attention and positional encoding ablation last logged validation | 35 |

Each source record carries a relative evidence path and SHA-256. The root aliases identify the legacy PE data tree and primary legacy PE checkout in the local archive manifest. Endpoints reflect the final available record in each source file. The archive does not infer successful completion or select a best run.

The public projection keeps numeric metrics, selected numeric settings and historical condition labels. It omits private supplementary comparisons, raw logs, raw data, weights, and operational details. Among the scanned families, 33 candidates were excluded as supplementary comparisons and 68 for unsupported schemas, settings or ambiguous validation sequences. The local evidence archive covers additional historical paths and variants.

A total of 313 rows have a recorded Git commit that resolves in legacy PE. The source files mark 76 of these rows as using modified code and 237 as clean. The other 280 rows have no recorded revision. All rows retain unverified exact execution provenance.
