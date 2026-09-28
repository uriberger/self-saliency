#!/usr/bin/env bash
#
# worktree.sh -- disposable git worktrees for change-work.
#
# The repo directory you launch sessions from (the "central tree") is the
# always-current reference copy: several sessions read it at once, so nothing
# ever edits it. Work that changes files happens in a throwaway worktree under
# .worktrees/, which is merged into main and deleted when it is done.
#
# new/done/drop must be run from the central tree -- they create and delete
# directories you would otherwise be standing in.
#
# Gitignored paths listed in .worktree-links are symlinked into every new
# worktree, so checkpoints/data/venvs are shared instead of re-created. They
# are shared *state*: writing to them from a worktree affects everyone.

set -euo pipefail

GIT_BIN=$(command -v git) || { echo "worktree.sh: git not found" >&2; exit 1; }

die()  { printf 'worktree.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

usage() {
    cat <<'EOF'
worktree.sh -- disposable git worktrees for change-work

  ./worktree.sh new <branch>           create .worktrees/<slug> on <branch>, off main
  ./worktree.sh list                   live worktrees, how far ahead, whether dirty
  ./worktree.sh done <branch> [--push] rebase onto main, fast-forward main, delete both
  ./worktree.sh drop <branch> [--force] delete branch + worktree, discarding the work
  ./worktree.sh sync                   fast-forward main from origin

Run new/done/drop from the central tree, not from inside a worktree.
EOF
    exit "${1:-0}"
}

# --- repo layout -------------------------------------------------------------

"$GIT_BIN" rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"

GIT_DIR=$(cd "$("$GIT_BIN" rev-parse --git-dir)" && pwd)
COMMON_DIR=$(cd "$("$GIT_BIN" rev-parse --git-common-dir)" && pwd)
CENTRAL=$(dirname "$COMMON_DIR")
WT_ROOT="$CENTRAL/.worktrees"
LINKS_FILE="$CENTRAL/.worktree-links"

# In a linked worktree the per-worktree git dir differs from the shared one.
if [ "$GIT_DIR" != "$COMMON_DIR" ]; then IN_WORKTREE=1; else IN_WORKTREE=0; fi

MAIN=$("$GIT_BIN" config worktree.main 2>/dev/null || true)
MAIN=${MAIN:-main}

# Every bare `git` in this script acts on the central tree, whatever the cwd.
git() { "$GIT_BIN" -C "$CENTRAL" "$@"; }

slug_of() { printf '%s' "${1//\//-}"; }

require_central() {
    [ "$IN_WORKTREE" -eq 0 ] || die "run this from the central tree:
    cd $CENTRAL; and ./worktree.sh $*"
}

# `git branch -d` measures "merged" against the central tree's HEAD, which is
# not necessarily $MAIN -- so check against $MAIN ourselves and force-delete.
delete_merged_branch() {
    git merge-base --is-ancestor "$1" "$MAIN" \
        || die "'$1' has commits that are not in $MAIN -- refusing to delete it"
    git branch -D "$1"
}

worktree_of_branch() {
    git worktree list --porcelain | awk -v b="branch refs/heads/$1" '
        /^worktree /{ p = substr($0, 10) }
        $0 == b     { print p; exit }'
}

fetch_origin() {
    git remote get-url origin >/dev/null 2>&1 || return 0
    if command -v timeout >/dev/null 2>&1; then
        timeout 30 "$GIT_BIN" -C "$CENTRAL" fetch --quiet origin "$MAIN" 2>/dev/null \
            || note "(could not reach origin -- working from the local $MAIN)"
    else
        git fetch --quiet origin "$MAIN" 2>/dev/null \
            || note "(could not reach origin -- working from the local $MAIN)"
    fi
}

# --- new ---------------------------------------------------------------------

link_shared() {
    local wt="$1" line src dst links_file="$wt/.worktree-links"
    # The branch's own list wins; fall back to the central tree's copy.
    [ -f "$links_file" ] || links_file="$LINKS_FILE"
    [ -f "$links_file" ] || { note "no .worktree-links -- nothing shared into the worktree"; return 0; }
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] || continue
        case "$line" in
            /*|*..*) note "skip (unsafe path): $line"; continue ;;
        esac
        src="$CENTRAL/$line"
        dst="$wt/$line"
        if [ ! -e "$src" ]; then
            note "skip (missing in the central tree): $line"
        elif [ -n "$(git ls-files -- "$line")" ]; then
            note "skip (tracked by git, comes with the checkout): $line"
        elif [ -e "$dst" ]; then
            note "skip (already in the worktree): $line"
        else
            mkdir -p "$(dirname "$dst")"
            ln -s "$src" "$dst"
            note "linked $line"
            # A symlink is a file to git, so a "foo/" pattern does not cover it.
            "$GIT_BIN" -C "$wt" check-ignore -q "$line" \
                || note "    WARNING: '$line' is not gitignored as a symlink -- add a slash-less '$line' to .gitignore, or it shows up as an untracked file"
        fi
    done < "$links_file"
}

cmd_new() {
    local branch="${1:-}" slug wt behind
    [ -n "$branch" ] || die "usage: ./worktree.sh new <branch>"
    require_central new "$branch"
    "$GIT_BIN" check-ref-format --branch "$branch" >/dev/null 2>&1 \
        || die "'$branch' is not a valid branch name"
    ! git show-ref --verify --quiet "refs/heads/$branch" \
        || die "branch '$branch' already exists"

    slug=$(slug_of "$branch")
    wt="$WT_ROOT/$slug"
    [ ! -e "$wt" ] || die "$wt already exists"

    fetch_origin
    if git show-ref --verify --quiet "refs/remotes/origin/$MAIN"; then
        behind=$(git rev-list --count "$MAIN..origin/$MAIN")
        [ "$behind" -eq 0 ] \
            || note "note: local $MAIN is $behind commit(s) behind origin/$MAIN -- run './worktree.sh sync' first if that matters"
    fi

    mkdir -p "$WT_ROOT"
    git worktree add -q "$wt" -b "$branch" "$MAIN"
    note "created $wt on '$branch', off $MAIN ($(git rev-parse --short "$MAIN"))"
    link_shared "$wt"

    printf '\nWork here (fish):\n    cd %s\n\nWhen done, from the central tree:\n    cd %s; and ./worktree.sh done %s\n' \
        "$wt" "$CENTRAL" "$branch"
}

# --- done --------------------------------------------------------------------

cmd_done() {
    local branch="${1:-}" push=0 arg slug wt ahead main_wt
    shift || true
    for arg in "$@"; do
        case "$arg" in
            --push) push=1 ;;
            *) die "unknown option: $arg" ;;
        esac
    done
    [ -n "$branch" ] || die "usage: ./worktree.sh done <branch> [--push]"
    require_central done "$branch"

    slug=$(slug_of "$branch")
    wt="$WT_ROOT/$slug"
    [ -d "$wt" ] || die "no worktree at $wt"
    git show-ref --verify --quiet "refs/heads/$branch" || die "no branch '$branch'"

    if [ -n "$("$GIT_BIN" -C "$wt" status --porcelain)" ]; then
        "$GIT_BIN" -C "$wt" status --short
        die "worktree has uncommitted changes -- commit or discard them first"
    fi

    ahead=$(git rev-list --count "$MAIN..$branch")
    [ "$ahead" -gt 0 ] \
        || die "'$branch' has no commits ahead of $MAIN -- use './worktree.sh drop $branch'"

    # Catch up with main so updating main is always a fast-forward.
    if ! git merge-base --is-ancestor "$MAIN" "$branch"; then
        if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
            note "'$branch' is published -- merging $MAIN into it instead of rebasing"
            "$GIT_BIN" -C "$wt" merge --no-edit "$MAIN" \
                || die "merge conflicts in $wt -- resolve them there, commit, then rerun"
        else
            note "rebasing '$branch' onto $MAIN"
            "$GIT_BIN" -C "$wt" rebase "$MAIN" || {
                "$GIT_BIN" -C "$wt" rebase --abort || true
                die "rebase hit conflicts -- run 'git rebase $MAIN' in $wt, resolve, then rerun"
            }
        fi
        ahead=$(git rev-list --count "$MAIN..$branch")
    fi

    main_wt=$(worktree_of_branch "$MAIN")
    if [ -n "$main_wt" ]; then
        "$GIT_BIN" -C "$main_wt" merge --ff-only "$branch" \
            || die "could not fast-forward $MAIN in $main_wt (uncommitted changes in the way?)"
    else
        git merge-base --is-ancestor "$MAIN" "$branch" \
            || die "refusing to move $MAIN non-fast-forward"
        git branch -f "$MAIN" "$branch"
        note "$MAIN moved to $(git rev-parse --short "$branch") (not checked out anywhere)"
    fi

    git worktree remove "$wt"
    delete_merged_branch "$branch"
    git worktree prune

    printf '\nMerged %s commit(s) into %s and removed %s\n' "$ahead" "$MAIN" "$wt"
    if [ "$push" -eq 1 ]; then
        git push origin "$MAIN"
    else
        printf 'Not pushed. To publish: git push origin %s\n' "$MAIN"
    fi
}

# --- drop --------------------------------------------------------------------

cmd_drop() {
    local branch="${1:-}" force=0 arg slug wt
    shift || true
    for arg in "$@"; do
        case "$arg" in
            --force|-f) force=1 ;;
            *) die "unknown option: $arg" ;;
        esac
    done
    [ -n "$branch" ] || die "usage: ./worktree.sh drop <branch> [--force]"
    require_central drop "$branch"

    slug=$(slug_of "$branch")
    wt="$WT_ROOT/$slug"

    if [ -d "$wt" ]; then
        if [ "$force" -eq 0 ] && [ -n "$("$GIT_BIN" -C "$wt" status --porcelain)" ]; then
            "$GIT_BIN" -C "$wt" status --short
            die "worktree has uncommitted changes -- rerun with --force to throw them away"
        fi
        if [ "$force" -eq 1 ]; then
            git worktree remove --force "$wt"
        else
            git worktree remove "$wt"
        fi
    fi

    if git show-ref --verify --quiet "refs/heads/$branch"; then
        if [ "$force" -eq 1 ]; then
            git branch -D "$branch"
        elif git merge-base --is-ancestor "$branch" "$MAIN"; then
            git branch -D "$branch"
        else
            die "'$branch' has commits that are not in $MAIN -- rerun with --force to throw them away"
        fi
    fi
    git worktree prune
    printf 'Dropped %s\n' "$branch"
}

# --- list / sync -------------------------------------------------------------

cmd_list() {
    local path branch ahead dirty
    git worktree list --porcelain | awk '
        /^worktree /{ p = substr($0, 10) }
        /^branch /  { print p "\t" substr($0, 8) }
        /^detached/ { print p "\tdetached" }' |
    while IFS=$'\t' read -r path branch; do
        branch="${branch#refs/heads/}"
        if [ "$path" = "$CENTRAL" ]; then
            printf 'central   %-30s %s\n' "$branch" "$path"
            continue
        fi
        ahead=$("$GIT_BIN" -C "$CENTRAL" rev-list --count "$MAIN..$branch" 2>/dev/null || echo '?')
        dirty=""
        [ -z "$("$GIT_BIN" -C "$path" status --porcelain 2>/dev/null)" ] || dirty="  [uncommitted changes]"
        printf 'worktree  %-30s +%s ahead of %s%s\n          %s\n' \
            "$branch" "$ahead" "$MAIN" "$dirty" "$path"
    done
}

cmd_sync() {
    local main_wt
    fetch_origin
    git show-ref --verify --quiet "refs/remotes/origin/$MAIN" || die "no origin/$MAIN to sync from"
    main_wt=$(worktree_of_branch "$MAIN")
    if [ -n "$main_wt" ]; then
        "$GIT_BIN" -C "$main_wt" merge --ff-only "origin/$MAIN" \
            || die "could not fast-forward $MAIN in $main_wt"
    else
        git fetch origin "$MAIN:$MAIN"
    fi
    printf '%s is at %s\n' "$MAIN" "$(git rev-parse --short "$MAIN")"
}

# --- dispatch ----------------------------------------------------------------

case "${1:-}" in
    new)  shift; cmd_new  "$@" ;;
    done) shift; cmd_done "$@" ;;
    drop) shift; cmd_drop "$@" ;;
    list) shift; cmd_list "$@" ;;
    sync) shift; cmd_sync "$@" ;;
    -h|--help|help|"") usage 0 ;;
    *) printf 'worktree.sh: unknown command: %s\n\n' "$1" >&2; usage 1 ;;
esac
