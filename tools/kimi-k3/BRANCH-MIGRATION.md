# Default branch: main

On 9 November 2026, this repository’s default branch was renamed from `kimi-k3` to `main`. This is a branch-name change, not an engine or model change. Release tags and commit hashes are unchanged.

## New installation

Follow the [installation guide](https://argonautlabs.ai/guides/install-kimi-k31.html), which pins a release tag. To work on the current development branch:

```sh
git clone --branch main https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1.git
```

## Existing clone

For a clone whose `origin` is this repository and whose local branch is still named `kimi-k3`, with no existing local `main` branch:

```sh
git branch -m kimi-k3 main
git fetch origin
git branch --set-upstream-to=origin/main main
git remote set-head origin -a
```

These commands rename the local branch and update tracking; they do not reset the working tree, discard local commits, or pull changes into it. If you use a different remote name, substitute it for `origin`. If a local `main` branch already exists, inspect your branches before renaming.

GitHub redirects normal web links using the old branch name. Raw-file URLs, scripts that request `kimi-k3`, and Git pulls do not follow that redirect: update those to `main`. Pinned release tags and commit URLs need no change.

[GitHub’s branch-renaming documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/renaming-a-branch).
