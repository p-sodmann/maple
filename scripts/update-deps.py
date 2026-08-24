#!/usr/bin/env python3
"""Check every third-party dependency in the workspace for a newer release.

Reads the version requirements out of the Cargo.toml files, asks the crates.io
sparse index what the newest non-yanked, non-prerelease version is, and compares
both against what Cargo.lock has actually locked.  Each dependency lands in one
of four buckets:

  current   locked version is the newest one published
  lock      a newer version already satisfies the requirement -> `cargo update`
  bump      the newest version is semver-incompatible -> the manifest must change
  pinned    the requirement is `=x.y.z` (or an operator this script won't guess
            at); left alone unless --include-pinned

With --apply the `bump` rows are rewritten in place (only the version literal on
the line is touched, so comments and formatting survive), then `cargo update`
runs, then `cargo check` verifies the workspace still builds.

Stdlib only; no cargo plugins to install.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

INDEX = "https://index.crates.io"
UA = "maple-update-deps (https://github.com/; cargo dependency check)"
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- semver bits

def parse_version(text):
    """'1.17.2-rc.1+meta' -> (1, 17, 2, ('rc', 1)).  Release sorts above pre."""
    text = text.split("+", 1)[0]
    core, _, pre = text.partition("-")
    parts = (core.split(".") + ["0", "0"])[:3]
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError:
        return None
    if not pre:
        # () sorts below any non-empty prerelease, so invert: release wins.
        return nums + (1, ())
    ident = tuple(int(p) if p.isdigit() else p for p in pre.split("."))
    return nums + (0, ident)


def version_key(v):
    # Mixed str/int prerelease identifiers can't be compared directly.
    return v[:4] + (tuple((0, p) if isinstance(p, int) else (1, p) for p in v[4]),)


def caret_satisfies(req_parts, latest):
    """Cargo's default caret rule: the leftmost non-zero component is locked."""
    req = [int(p) for p in req_parts]
    lo = (req + [0, 0])[:3]
    if latest[:3] < tuple(lo):
        return False
    if req[0] != 0:
        return latest[0] == req[0]
    if len(req) == 1:                       # "0" -> >=0.0.0, <1.0.0
        return latest[0] == 0
    if req[1] != 0 or len(req) == 2:        # "0.25" / "0.25.1" -> <0.26.0
        return latest[:2] == (0, req[1])
    return latest[:3] == (0, 0, req[2])     # "0.0.3" -> <0.0.4


# ---------------------------------------------------------------- manifests

HEADER_RE = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(?:#.*)?$")
ENTRY_RE = re.compile(r"^\s*(?:(['\"])(?P<q>[^'\"]+)\1|(?P<b>[A-Za-z0-9_.+-]+))\s*=\s*(?P<val>.+?)\s*$")
INLINE_VERSION_RE = re.compile(r"version\s*=\s*\"([^\"]+)\"")
STRING_RE = re.compile(r"^\"([^\"]+)\"")


def manifests():
    found = [os.path.join(ROOT, "Cargo.toml")]
    for entry in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, entry, "Cargo.toml")
        if os.path.isfile(path):
            found.append(path)
    return found


def table_kind(header):
    """-> ('table', None) for a dep table, ('entry', name) for [dependencies.x]."""
    parts = header.split(".")
    if parts[-1] in DEP_TABLES:
        return "table", None
    if len(parts) >= 2 and parts[-2] in DEP_TABLES:
        return "entry", parts[-1].strip("'\"")
    return None, None


def scan(path):
    """Yield every third-party requirement as a dict, with its source line."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    out = []
    kind, sub = None, None
    for lineno, line in enumerate(lines):
        header = HEADER_RE.match(line)
        if header:
            kind, sub = table_kind(header.group(1))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if kind == "entry":
            m = INLINE_VERSION_RE.search(line)
            if m and stripped.startswith("version"):
                out.append(dict(name=sub, req=m.group(1), path=path, lineno=lineno,
                                literal=m.group(1)))
            continue
        if kind != "table":
            continue

        m = ENTRY_RE.match(line)
        if not m:
            continue
        name = m.group("q") or m.group("b")
        val = m.group("val")
        if val.startswith("{"):
            if "path" in val or "git" in val or re.search(r"workspace\s*=\s*true", val):
                continue
            v = INLINE_VERSION_RE.search(val)
            if not v:
                continue      # e.g. an optional dep with only `features`
            req = v.group(1)
        else:
            s = STRING_RE.match(val)
            if not s:
                continue
            req = s.group(1)
        out.append(dict(name=name, req=req, path=path, lineno=lineno, literal=req))
    return out


def locked_versions():
    lock = os.path.join(ROOT, "Cargo.lock")
    if not os.path.isfile(lock):
        return {}
    versions, name = {}, None
    with open(lock, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line == "[[package]]":
                name = None
            elif line.startswith("name = "):
                name = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("version = ") and name:
                versions.setdefault(name, []).append(line.split("=", 1)[1].strip().strip('"'))
    return versions


# ---------------------------------------------------------------- crates.io

def index_path(name):
    n = name.lower()
    if len(n) == 1:
        return "1/" + n
    if len(n) == 2:
        return "2/" + n
    if len(n) == 3:
        return "3/%s/%s" % (n[0], n)
    return "%s/%s/%s" % (n[:2], n[2:4], n)


def latest_release(name, allow_pre):
    """-> ((version_str, parsed), None) | (None, error).  Falls back to
    prereleases when a crate has published nothing else (ort's 2.0.0-rc.N)."""
    url = "%s/%s" % (INDEX, index_path(name))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return None, "not on crates.io" if e.code == 404 else "http %d" % e.code
    except Exception as e:                                # network, DNS, TLS…
        return None, str(e)

    best = None
    for line in body.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("yanked"):
            continue
        v = parse_version(rec["vers"])
        if v is None or (v[3] == 0 and not allow_pre):
            continue
        if best is None or version_key(v) > version_key(best[1]):
            best = (rec["vers"], v)
    if best is None:
        if allow_pre:
            return None, "no releases"
        return latest_release(name, True)
    return best, None


# ---------------------------------------------------------------- reporting

COLORS = {"current": "\033[32m", "lock": "\033[33m", "bump": "\033[31m",
          "pinned": "\033[90m", "error": "\033[35m"}
RESET = "\033[0m"


def paint(text, status, on):
    return "%s%s%s" % (COLORS[status], text, RESET) if on else text


def classify(dep, latest_str, latest):
    req = dep["req"].strip()
    if req.startswith("="):
        return "pinned"
    if not re.match(r"^\^?\d", req):
        return "pinned"            # >=, ~, *, multi-clause — don't guess
    parts = re.match(r"^\^?([\d.]+)", req).group(1).rstrip(".").split(".")
    return "lock" if caret_satisfies(parts, latest) else "bump"


def pick_locked(dep, versions):
    """Cargo.lock can hold several copies of a crate; keep the one our own
    requirement resolved to, so the column answers "what are *we* using"."""
    req = dep["req"].strip().lstrip("=^")
    m = re.match(r"^([\d.]+)", req)
    if m and len(versions) > 1:
        parts = m.group(1).rstrip(".").split(".")
        mine = [v for v in versions
                if parse_version(v) and caret_satisfies(parts, parse_version(v))]
        if mine:
            return sorted(mine, key=lambda v: version_key(parse_version(v)))[-1:]
    return versions


def new_literal(dep, latest_str, latest):
    """Keep the requirement's precision: '0.17' + 0.18.1 -> '0.18'."""
    req = dep["req"].strip()
    prefix = "^" if req.startswith("^") else ""
    depth = len(re.match(r"^\^?([\d.]+)", req).group(1).rstrip(".").split("."))
    if "-" in latest_str or "+" in latest_str:
        return prefix + latest_str
    return prefix + ".".join(latest_str.split(".")[:depth])


def rewrite(dep, literal):
    with open(dep["path"], encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)
    line = lines[dep["lineno"]]
    old = '"%s"' % dep["literal"]
    if old not in line:
        return False
    lines[dep["lineno"]] = line.replace(old, '"%s"' % literal, 1)
    with open(dep["path"], "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return True


def run(cmd):
    print("\n$ %s" % " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite outdated requirements, then cargo update + cargo check")
    ap.add_argument("--lock-only", action="store_true",
                    help="with --apply: run cargo update but leave the manifests alone")
    ap.add_argument("--no-verify", action="store_true",
                    help="with --apply: skip the cargo check afterwards")
    ap.add_argument("--include-pinned", action="store_true",
                    help="also report (and with --apply, bump) `=x.y.z` requirements")
    ap.add_argument("--allow-prerelease", action="store_true",
                    help="consider prerelease versions as update candidates")
    ap.add_argument("--only", nargs="+", metavar="CRATE",
                    help="restrict to these crates")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    color = not args.no_color and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

    deps, seen = [], {}
    for path in manifests():
        for dep in scan(path):
            if args.only and dep["name"] not in args.only:
                continue
            key = (dep["name"], dep["req"])
            if key in seen:          # same requirement in two manifests
                seen[key].append(dep)
                continue
            seen[key] = [dep]
            deps.append(dep)
    if not deps:
        print("no third-party dependencies found")
        return 0

    locked = locked_versions()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(
            lambda d: latest_release(
                d["name"], args.allow_prerelease or "-" in d["req"]), deps))

    rows, buckets = [], {"current": [], "lock": [], "bump": [], "pinned": [], "error": []}
    for dep, (best, err) in zip(deps, results):
        if err:
            rows.append((dep, "error", "-", err))
            buckets["error"].append(dep)
            continue
        latest_str, latest = best
        status = classify(dep, latest_str, latest)
        if status == "pinned" and not args.include_pinned:
            rows.append((dep, "pinned", latest_str, "pinned requirement"))
            buckets["pinned"].append(dep)
            continue
        if status == "pinned":
            status = "bump" if dep["req"].lstrip("=").strip() != latest_str else "current"
        here = pick_locked(dep, sorted(set(locked.get(dep["name"], []))))
        if status == "lock" and here and all(parse_version(v)[:3] == latest[:3] for v in here):
            status = "current"
        rows.append((dep, status, latest_str, ", ".join(here) or "-"))
        buckets[status].append(dep)

    width = max(len(d["name"]) for d, _, _, _ in rows)
    reqw = max(8, max(len(d["req"]) for d, _, _, _ in rows))
    lockw = max(6, max(len(n) for _, s_, _, n in rows if s_ not in ("error", "pinned")) if
                any(s_ not in ("error", "pinned") for _, s_, _, n in rows) else 6)
    latw = max(6, max(len(l) for _, _, l, _ in rows))
    order = {"bump": 0, "lock": 1, "error": 2, "pinned": 3, "current": 4}
    print("%-*s  %-*s  %-*s  %-*s  %s" % (width, "CRATE", reqw, "REQUIRES",
                                          lockw, "LOCKED", latw, "LATEST", "STATUS"))
    for dep, status, latest_str, note in sorted(rows, key=lambda r: (order[r[1]], r[0]["name"])):
        locked_col = note if status not in ("error", "pinned") else "-"
        print("%-*s  %-*s  %-*s  %-*s  %s" % (
            width, dep["name"], reqw, dep["req"], lockw, locked_col, latw, latest_str,
            paint(status if status != "pinned" else "pinned (skipped)", status, color)))
        if status == "error":
            print("%-*s  %s" % (width, "", note))

    print("\n%d up to date, %d lockfile update%s, %d manifest bump%s, %d pinned, %d unreachable" % (
        len(buckets["current"]), len(buckets["lock"]), "" if len(buckets["lock"]) == 1 else "s",
        len(buckets["bump"]), "" if len(buckets["bump"]) == 1 else "s",
        len(buckets["pinned"]), len(buckets["error"])))

    if not args.apply:
        if buckets["bump"] or buckets["lock"]:
            print("\nre-run with --apply to update")
        return 0

    changed = []
    if not args.lock_only:
        for dep, status, latest_str, _ in rows:
            if status != "bump":
                continue
            literal = new_literal(dep, latest_str, parse_version(latest_str))
            for d in seen[(dep["name"], dep["req"])]:
                if rewrite(d, literal):
                    rel = os.path.relpath(d["path"], ROOT)
                    print("  %s: %s %s -> %s" % (rel, d["name"], d["literal"], literal))
                    changed.append(d["name"])
                else:
                    print("  ! could not rewrite %s in %s:%d — edit it by hand"
                          % (d["name"], os.path.relpath(d["path"], ROOT), d["lineno"] + 1))

    if run(["cargo", "update"]) != 0:
        print("\ncargo update failed — manifests were left edited; `git diff` to review")
        return 1
    if args.no_verify:
        return 0
    code = run(["cargo", "check", "--workspace", "--all-targets"])
    if code != 0:
        print("\ncargo check failed after updating%s.\n"
              "Review with `git diff` and `git checkout -- '*Cargo.toml' Cargo.lock` to undo."
              % (" " + ", ".join(sorted(set(changed))) if changed else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
