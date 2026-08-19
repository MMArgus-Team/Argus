#!/usr/bin/env node
/**
 * stage-hermes-source.cjs
 *
 * Snapshots the git-tracked Hermes agent source into
 * apps/desktop/build/hermes-src/, so a SELF-CONTAINED desktop build can ship
 * its own agent source (electron-builder extraResources → resources/hermes-src)
 * and bootstrap from it OFFLINE, instead of cloning a (possibly unpushed) commit
 * from GitHub. See bootstrap-runner.cjs (bundledSourceRoot) + install.ps1
 * (-LocalSource).
 *
 * Source list = `git ls-files` (tracked files), copied from the WORKING TREE.
 * This ships exactly what's on disk — INCLUDING uncommitted local changes — so a
 * local dev build is self-contained with your current code, without forcing a
 * commit first. It's complete (all files install.ps1 needs: pyproject/uv.lock,
 * agent/, tools/, hermes_cli/, scripts/, …) and auto-excludes runtime junk
 * (node_modules, .venv, caches, logs, egg-info, __pycache__) since those are
 * gitignored / untracked. No hand-maintained allow/deny list to drift.
 *
 * Idempotent: wipes + rewrites build/hermes-src each run.
 */
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const DESKTOP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const OUT_DIR = path.join(DESKTOP_ROOT, 'build', 'hermes-src')

function isGitRepo() {
  try {
    execFileSync('git', ['rev-parse', '--is-inside-work-tree'], {
      cwd: REPO_ROOT,
      stdio: 'ignore'
    })
    return true
  } catch {
    return false
  }
}

function main() {
  if (!isGitRepo()) {
    // Not fatal: a build from a non-git source tree just won't ship bundled
    // source (bootstrap falls back to its normal GitHub/stamp path). Warn loud.
    console.warn(
      '[stage-hermes-source] WARNING: repo root is not a git work tree; ' +
        'skipping bundled-source snapshot. The packaged app will NOT be ' +
        'self-contained and will try to fetch install from GitHub.'
    )
    return
  }

  // Fresh output dir.
  fs.rmSync(OUT_DIR, { recursive: true, force: true })
  fs.mkdirSync(OUT_DIR, { recursive: true })

  console.log(`[stage-hermes-source] copying tracked source (working tree) → ${path.relative(REPO_ROOT, OUT_DIR)}`)
  // List all tracked files (NUL-delimited to be safe with any path). This is the
  // tracked set; copying from the working tree captures uncommitted edits too.
  const listed = execFileSync('git', ['ls-files'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 1024
  })
  // Newline-delimited tracked paths. (Paths containing newlines are not used
  // in this repo; git would quote them and they'd fail the copy below, which is
  // acceptable — none exist.)
  const rels = listed.split('\n').map(x => x.trim()).filter(Boolean)
  let copied = 0
  for (const rel of rels) {
    const src = path.join(REPO_ROOT, rel)
    // A tracked file can be absent on disk (e.g. deleted-but-staged); skip it.
    let st
    try {
      st = fs.lstatSync(src)
    } catch {
      continue
    }
    if (!st.isFile() && !st.isSymbolicLink()) continue
    const dest = path.join(OUT_DIR, rel)
    fs.mkdirSync(path.dirname(dest), { recursive: true })
    fs.copyFileSync(src, dest)
    copied++
  }
  console.log(`[stage-hermes-source] copied ${copied} files`)

  // Sanity: the installer + pyproject must be present, else the snapshot is
  // useless and we should fail the build loudly rather than ship a broken app.
  const mustExist = ['scripts/install.ps1', 'pyproject.toml']
  const missing = mustExist.filter(rel => !fs.existsSync(path.join(OUT_DIR, rel)))
  if (missing.length) {
    throw new Error(
      `[stage-hermes-source] snapshot missing required files: ${missing.join(', ')}. ` +
        'Bundled-source bootstrap would fail.'
    )
  }
  console.log('[stage-hermes-source] done')
}

main()
