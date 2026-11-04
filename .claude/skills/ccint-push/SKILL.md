---
name: ccint-push
description: Where and how to commit and push work done in ~/jingrui (ccint). Use whenever you are about to commit, push, open a PR, add a remote, or publish any file from this repo, or when someone asks where ccint code or docs should go.
---

# Pushing ccint (`~/jingrui`)

All work in `~/jingrui` goes to **`astronautrronearth/ccint`** (remote `origin`). No other destination.

- Push to `origin`, to the branch you are working on (the current research branch is `interaction-experiment`; `main` only through a PR).
- Do not add other remotes, and do not push this repo, its code or its docs to any other account or repo, including `W4ng77/*`.
- `www2027/` is a local, excluded folder (`.git/info/exclude`) holding a separate document repo. Leave it out of ccint commits and do not push it unless the owner asks for that one push.

## `astronautrronearth/ccint` is a public repo

Before every push, check what is going out (`git log origin/<branch>..HEAD --stat`) and keep these out:
- `.env`, `.bsky_session.json`, any key, token or app password (only `.env.example`, with names and no values);
- `pgdata/`, `outputs/`, `logs/`, `reports/`: generated, and the annotation set and eval sets contain Bluesky handles and post text;
- `eval/canada_screen*.tsv` (handles and text, see `eval/README.md`);
- submission plans and drafts: `WWW2027_PROPOSAL.md`, `WWW2027_SHORT_PAPER.md` and other paper drafts (the venue is double-blind).

Most of these are already in `.gitignore`; never force-add them (`git add -f`).

## Before pushing

1. `git status` and `git log origin/<branch>..HEAD --stat`: only intended files.
2. Run the tests that cover what changed (`.venv/bin/python -m pytest tests/...`).
3. Ask the owner before the push. Never force-push or rewrite history that is already on `origin`.
