# Canonical Source — `src/mlBridge/`

This directory is the **single canonical source** for the `mlBridge` Python package
used by every active project in this workspace (ACBL pipeline, Elo Ratings app,
FFBridge app, BBO bidding tools, etc.).

## How sharing works

Other subprojects do not vendor copies of these files. Instead they create a
Windows junction back to this directory using each project's `mklinks.bat`,
typically:

```bat
mklink /J mlBridge ..\..\mlBridge
```

The subproject's `sys.path.append(... / 'mlBridge')` then resolves to this
canonical directory. Imports look like `from mlBridge.mlBridgeAugmentLib import ...`.

## "Duplicate" copies you may see in the workspace

You will find files with the same name under at least two other paths:

- `src/ffbridge/ffbridge-postmortem/src/mlbridgelib/mlBridge/`
- `src/github-tests/ffbridge-postmortem/mlBridge/`

Both of these directories live **inside other, independent git repositories**
(`src/ffbridge/ffbridge-postmortem/.git`, `src/github-tests/ffbridge-postmortem/.git`).
They are vendored snapshots owned by those external repos, not by this workspace.

**Do not edit them.** They are out-of-date and not imported by any active code
path here — `ffbridge_streamlit.py` (the live one) imports from the junction
created by `mklinks.bat`, which points back to *this* directory.

If you need to refresh those vendored copies, do it inside their own git repos
as a separate exercise.

## Rule of thumb

If you are editing a file in `mlBridge/...`, the file you should be editing is
the one under `src/mlBridge/`. Nothing else.
