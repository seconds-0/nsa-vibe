Reports Directory

Purpose and Organization:
- Centralize all Test Engineer and Core Engineer reports for easy discovery.

Conventions:
- Location: All reports live under `Documentation/Reports/`.
- Naming:
  - Test Engineer (Claude): `<yyyy-mm-dd> Test Engineer Report - <Subject> <vX>`
  - Core Engineer (GPT‑5): `<yyyy-mm-dd> Core Engineer Report - <Subject> <vX>`
- Examples:
  - `2025-08-26 Test Engineer Report - DDP One-Step Trace v1.md`
  - `2025-08-26 Core Engineer Report - GC Remediation v2.md`

Guidelines:
- Keep one primary markdown per subject per day when possible; use version suffixes (`v2`, `v3`) for iterations.
- Attach raw logs/CSV under `artifacts/` and link from the report instead of inlining large blobs.
- Avoid placing reports in the repo root; use this directory.

