# SonarQube Cloud CI setup

This repository uses CI-based analysis so SonarQube Cloud can import the
coverage.py report produced by the Python 3.12 test job.

Before enabling the `sonar` job, a project administrator must:

1. Open **Aisha → Administration → Analysis Method** in SonarQube Cloud and
   disable **Automatic Analysis**. Automatic Analysis cannot import coverage
   reports, and it must not run alongside this CI integration.
2. Add a GitHub Actions `SONAR_TOKEN` secret with **Execute Analysis**
   permission for the `gearbox_aisha` project.

The workflow intentionally skips the Sonar job for pull requests from forks:
GitHub does not make repository secrets available to them. Linting and tests
still run for every pull request.
