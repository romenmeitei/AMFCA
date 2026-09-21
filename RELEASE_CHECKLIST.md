# Public-release checklist

- [ ] Replace author placeholders in `CITATION.cff.example` and rename to `CITATION.cff`.
- [ ] Confirm the MIT license is the intended code license.
- [ ] Replace `YOUR_USERNAME` in the master Colab repository URL.
- [ ] Review `configs/lhende_2026.yaml` for any account-specific or private paths before publication.
- [ ] Confirm Earth Engine project identifiers are intended to be public; otherwise replace with a placeholder and document user configuration.
- [ ] Run `python scripts/verify_repository.py`.
- [ ] Confirm GitHub Actions passes.
- [ ] Confirm no copyrighted/redistribution-restricted imagery is committed.
- [ ] Add manuscript DOI/preprint DOI when available.
- [ ] Tag release `v1.0.0`.
