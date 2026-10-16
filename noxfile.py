#!/usr/bin/env -S uv run -q

# /// script
# dependencies = ["nox>=2026.8.17"]
# ///

import nox

nox.needs_version = ">=2026.8.17"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run all pre-commit checks."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-r", "requirements.txt", "pytest")
    session.run("pytest", *session.posargs)


if __name__ == "__main__":
    nox.main()
