# Contributing to TADS-X

Thank you for your interest in contributing to **TADS-X**! 🎉

This document outlines the guidelines for reporting issues, proposing changes, and submitting pull requests.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Commit Message Style](#commit-message-style)
- [Pull Request Checklist](#pull-request-checklist)
- [Code Style](#code-style)

---

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How to Report a Bug

1. Search the [existing issues](https://github.com/kishlabs/TADS-X/issues) to avoid duplicates.
2. Open a [Bug Report](https://github.com/kishlabs/TADS-X/issues/new?template=bug_report.md) and fill in all requested fields.
3. Include the output of `python --version` and `pip list` alongside any stack traces.

---

## How to Request a Feature

1. Open a [Feature Request](https://github.com/kishlabs/TADS-X/issues/new?template=feature_request.md).
2. Describe the motivation, expected behaviour, and potential implementation approach.

---

## Development Setup

```bash
git clone https://github.com/kishlabs/TADS-X.git
cd TADS-X

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Branch Naming Conventions

| Type        | Pattern                       | Example                      |
|-------------|-------------------------------|------------------------------|
| Feature     | `feat/<short-description>`    | `feat/agca-refactor`         |
| Bug fix     | `fix/<short-description>`     | `fix/scrn-nan-loss`          |
| Docs        | `docs/<short-description>`    | `docs/update-readme`         |
| Chore       | `chore/<short-description>`   | `chore/update-dependencies`  |

---

## Commit Message Style

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

---

## Pull Request Checklist

Before opening a PR, ensure:

- [ ] The code passes `flake8` linting (`make lint`)
- [ ] All new functionality is documented (docstrings + README if user-facing)
- [ ] You have updated `CHANGELOG.md` under `[Unreleased]`
- [ ] You have added or updated relevant tests (if applicable)
- [ ] The PR description explains *what* and *why*, not just *how*

---

## Code Style

- Follow [PEP 8](https://pep8.org/).
- Maximum line length: **120 characters**.
- Use type hints where practical.
- All public functions and classes **must** have docstrings.
