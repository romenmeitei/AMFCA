# GitHub upload steps

1. Create an empty GitHub repository named `AMFCA` (do not add a README or license on GitHub because both are already included here).
2. From the extracted repository folder:

```bash
git init
git add .
git commit -m "AMFCA v1.0.0 reproducibility release"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/AMFCA.git
git push -u origin main
```

3. Update `CITATION.cff.example` with author metadata and repository URL, rename it to `CITATION.cff`, commit, and push.
4. The master Colab is configured to clone the official AMFCA repository at `https://github.com/romenmeitei/AMFCA.git`.
5. Confirm the GitHub Actions `tests` workflow passes.
6. Create a tagged release `v1.0.0` and attach the release ZIP/checksum if desired.
7. If the manuscript has a DOI or archived release, add it to README/CITATION before submission.

Do not commit third-party satellite rasters unless their licenses explicitly permit redistribution.
