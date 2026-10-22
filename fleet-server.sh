#!/bin/sh
# fleet-server.sh — single switcher for the ds4 model fleet.
#
# RUNTIME CONTRACT: the fleet does NOT run under launchd. The switcher
# generates a per-model launch script (~/ds4/fleet/<label>.launch.sh),
# halts whatever is resident, boots the server as a plain detached
# background process (nohup /bin/sh <launch> &), and terminates. It opens
# NO terminal, session, or window — it loads the model and exits. The
# launch script reproduces KeepAlive(SuccessfulExit=false,
# ThrottleInterval=4) as a shell loop: nonzero exit relaunches after 4s,
# exit 0 stays down, a stop flag (~/ds4/fleet/<label>.stop) halts it.
# Hand-halt = plant the flag BEFORE SIGTERM.
#
# Five models, one resident at a time (the engine's instance lock):
#   glm53flash  GLM 5.3 Flash Q4_K  (workhorse)        ~/ds4         port 8000
#   glm53full   GLM 5.3 full build B (ctx 440k)        ~/ds4-glmfull port 8008
#   ds41f       DeepSeek V4.1 Flash 1017 IQ2/Q2 imat   ~/ds4-v41     port 8005
#   muse        Muse Spark Open 816B textbase (1M ctx) ~/ds4-muse    port 8123
#   qwen38fn    Qwen3.8 Flash Next Q4 (stable)         ~/ds4-qfn     port 8007
#
# ds4f (vanilla V4 Flash, ~/ds4-baseline) is SUPERSEDED: dropped from the
# switcher and the purge table. The baseline worktree stays the ops tree.
#
# LOGGING: the detached loop's console output (weight-load progress,
# server log) streams into the LAUNCHING TERMINAL PANE (the launch script
# attaches via `exec >/dev/tty` inside itself — nohup with stdout already
# redirected to /dev/null is tty-less, and the probe subshell keeps a
# tty-less launch like cron/ssh -T from dying on the redirect). Output
# ALWAYS also tees to the per-model log ~/Library/Logs/ds4-fleet/<model>.log
# (non-dated; rotate-logs.sh copytruncate target). `fleet-server.sh log
# [model]` tails it. --debug (or DS4_FLEET_DEBUG=1) additionally arms a
# PER-LOAD artifact <model>-<YYYYmmdd-HHMMSS>.log — use it for
# measurement/era-match windows so the binary_sha256 banner lands in a
# self-contained file; dated files are outside rotate-logs.sh.
#
# No RunAtLoad: nothing auto-starts at login anymore — the operator boots
# the workhorse after a reboot (`fleet-server.sh glm53flash`), or may
# install ONE RunAtLoad LaunchAgent whose ProgramArguments is exactly that
# one-shot command. Never a second server in LaunchAgents.
#
# ENV SUBSTITUTION — the launch-script boundary: the launch script is a NEW
# shell and never inherits the caller's environment, so `DS4_FOO=1
# fleet-server.sh <model>` is silently DROPPED and any verdict from such a
# load is invalid. Supported channels, all rendered INTO the launch script
# at generation time (audit it there — never trust the caller line, and
# never audit a running load via `ps eww`, which macOS truncates):
#   1. per-model caller knobs (GLM53FULL_*, see the glm53full case);
#   2. DS4_FLEET_ENVS="K=V ..." generic passthrough, any model;
#   3. the arm queue — rendered last, so an arm always wins.
#
# ARM QUEUE — cursor-driven A/B cycling, generalized to every model:
#   ~/ds4/fleet/arms-<model>   definitions: <name><TAB>KEY=VALUE KEY=VALUE ...
#                              (blank lines and #-comments ignored)
#   ~/ds4/fleet/arm-<model>    cursor: the arm the NEXT load serves.
# A real load renders the cursor arm and ADVANCES the cursor; past the
# last arm the cursor is deleted (terminal off = standing production).
# DS4_FLEET_ARM=<name> serves one arm WITHOUT advancing. DRYRUN never
# touches the cursor. A cursor naming an undefined arm is a HARD ERROR —
# never silently serve the unpatched default.
#
# INSTANCE-LOCK FILE LIFECYCLE — the switcher owns it; the operator never
# touches /tmp by hand. The engine creates the lock (O_CREAT) and never
# unlinks it; only a live flock holder blocks a load. Therefore halt order
# is load-bearing:
#   1. plant stop flags for every model;
#   2. kill the launch LOOPS FIRST — a loop sitting in its throttle sleep
#      wakes and execs a NEW server without re-checking the flag (the flag
#      is only checked after a server exit); this is the race that used to
#      re-create the lock right after every deletion;
#   3. SIGTERM every ds4-server AND every live lock holder found via lsof
#      (catches engine processes under any binary name: ds4 CLI, tests,
#      benches — pgrep alone misses them), and boot out any legacy launchd
#      job (a stale KeepAlive job would restart-loop against this runtime);
#   4. wait until nothing is alive (SIGTERM only — NEVER kill -9: the
#      ledger/atexit work dies with the process);
#   5. delete the lock file (pure debris at that point; the next engine
#      start re-creates it).
# A load refuses to boot while the lock still has a live holder and prints
# the offending pids instead of letting the engine die on it.
#
# SSD cold KV is OFF by default; --kv adds --kv-disk-dir/--kv-disk-space-mb
# (ds41f wires it standing — it is the cache-miss fix, not an option). The
# KV_SPACE_MB_* budgets below sum to the 1.5 TiB fleet envelope and are
# read back by ~/ds4/purge-kv-caches.sh (sed on line-start
# KV_SPACE_MB_<KEY>=) — keep the lines column-1 and the sum exact.
#
# BIND PREFLIGHT: the switcher refuses to launch unless HOST is assigned
# to a local interface (or 0.0.0.0/127.0.0.1). A wrong bind address used
# to pass every gate (--help parses before bind) and cost full ~180 GiB
# weight mappings per crash-loop round before dying at listen().
#
# Usage:
#   fleet-server.sh <glm53flash|glm53full|ds41f|muse|qwen38fn> [--kv] [--debug]
#   fleet-server.sh status
#   fleet-server.sh log [model]
#   fleet-server.sh stop
#   DS4_FLEET_DRYRUN=1 ...   # write launch script + preview, touch nothing
#   DS4_FLEET_SLEEP=<sec>    # minimum settle sleep after the stop (default 8)
#   DS4_FLEET_DEBUG=1        # same as --debug
#   DS4_FLEET_ENVS="K=V ..." # generic env substitution
#   DS4_FLEET_ARM=<name>     # serve one named arm, cursor unchanged
#   DS4_LOCK_FILE=<path>     # engine instance-lock path (default /tmp/ds4.lock);
#                            # exported into every launch script so stop/load
#                            # manage exactly the file the engine uses
#
# Pitfalls encoded here (ds4-server-fleet-ops skill):
#   - pgrep -x, never pgrep -f ("less ds4-server.log" must not match).
#   - "Launched" means the detached loop is running — weights take 45-90s
#     to map; the HTTP port answers later.
#   - A crash-restart loop is a loop: after rebuilding to fix a crash, the
#     loop relaunches the NEW binary bytes on its next attempt.
#   - To audit a running load's env, read its launch script — the single
#     source of truth for that load. Never ps.

set -eu

FLEET_DIR="$HOME/ds4/fleet"
LOGDIR="$HOME/Library/Logs/ds4-fleet"
MIN_SLEEP="${DS4_FLEET_SLEEP:-8}"
DRYRUN="${DS4_FLEET_DRYRUN:-0}"
DEBUG="${DS4_FLEET_DEBUG:-0}"
WAIT_TIMEOUT=180
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4.lock}"

# Shared fleet geometry (per-model TOKENS override wins where set).
THREADS=24
POWER=100
HOST=192.168.1.232
TOKENS=384000
THROTTLE=4

# Per-model SSD KV budgets (MB) — column-1 lines, parsed by
# purge-kv-caches.sh; the five sum to 1572864 MB = the 1.5 TiB envelope.
# Keep it that way. (MUSE holds the retired DS4F slot's 196608.)
KV_SPACE_MB_DS41F=65536
KV_SPACE_MB_GLM53FLASH=1048576
KV_SPACE_MB_GLM53FULL=65536
KV_SPACE_MB_MUSE=196608
KV_SPACE_MB_QWEN38FN=196608

MODELS="glm53flash glm53full ds41f muse qwen38fn"

# Legacy launchd labels — boot-out only, transition safety: a stale
# KeepAlive job restart-loops against this runtime if a plain kill meets
# it. Nothing here loads plists anymore.
ALL_LABELS="com.ds4-glm5.3-flash.server com.ds4-glm53-full.server com.ds4-ds4f.server com.ds4-v41.server com.ds4-qfn.server"
GUI="gui/$(id -u)"

die() { echo "ERROR: $*" >&2; exit 1; }

do_run() {
    if [ "$DRYRUN" = "1" ]; then echo "DRYRUN + $*"; else "$@"; fi
}

label_for() {
    case "$1" in
    glm53flash) echo com.ds4-glm5.3-flash.server ;;
    glm53full)  echo com.ds4-glm53-full.server ;;
    ds41f)      echo com.ds4-v41.server ;;
    muse)       echo com.ds4-muse.server ;;
    qwen38fn)   echo com.ds4-qfn.server ;;
    esac
}

lock_holders() {
    lsof -t "$LOCK_FILE" 2>/dev/null || true
}

# The bind preflight: refuse to launch to an address this machine does not
# own. --help cannot catch a bad --host (parsing succeeds; the failure
# lands at listen(), after the full weight mapping).
host_bindable() {
    case "$HOST" in
    0.0.0.0|127.0.0.1|localhost) return 0 ;;
    esac
    ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -qx "$HOST"
}

# ---- status ----------------------------------------------------------------
status_all() {
    echo "fleet launch loops (pid files):"
    found=0
    for m in $MODELS; do
        L=$(label_for "$m")
        [ -f "$FLEET_DIR/$L.loop" ] || continue
        pid=$(cat "$FLEET_DIR/$L.loop" 2>/dev/null || true)
        [ -n "$pid" ] || continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  $m: launch loop pid $pid alive"
            found=1
        fi
    done
    [ "$found" = "1" ] || echo "  none alive"
    echo "processes:"
    pgrep -lx ds4-server || echo "  no ds4-server process"
    echo "instance lock ($LOCK_FILE):"
    if [ -e "$LOCK_FILE" ]; then
        _h=$(lock_holders | tr '\n' ' ')
        if [ -n "$_h" ]; then
            echo "  LIVE holder(s): $_h (stop will halt them)"
        else
            echo "  file present, NO live holder — stale debris; stop/load deletes it"
        fi
    else
        echo "  absent"
    fi
    echo "listening ports:"
    lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '/ds4-serve/ {print "  pid "$2" -> "$9}' || true
    echo "logs (newest 5):"
    ls -t "$LOGDIR"/*.log 2>/dev/null | head -5 | sed 's/^/  /' || echo "  none"
    echo "legacy launchd jobs (should be none):"
    found=0
    for L in $ALL_LABELS; do
        if launchctl print "$GUI/$L" >/dev/null 2>&1; then
            echo "  $L STILL LOADED — stop will boot it out"
            found=1
        fi
    done
    [ "$found" = "1" ] || echo "  none loaded"
    echo "arm queues:"
    for m in $MODELS; do
        cur=$(cat "$FLEET_DIR/arm-$m" 2>/dev/null || true)
        if [ -n "$cur" ]; then
            echo "  $m: cursor=$cur (defs: $FLEET_DIR/arms-$m)"
        fi
    done
    echo "  (models without a cursor line serve standing production)"
}

# ---- shared halt -------------------------------------------------------------
# See the lock-lifecycle block in the header for why the order is: flags,
# loops, legacy jobs, then servers+holders, wait, delete the lock.
# Returns 1 if engine processes outlive WAIT_TIMEOUT (lock NOT deleted).
kill_running() {
    for m in $MODELS; do
        [ "$DRYRUN" = "1" ] || : > "$FLEET_DIR/$(label_for "$m").stop"
    done
    for m in $MODELS; do
        L=$(label_for "$m")
        [ -f "$FLEET_DIR/$L.loop" ] || continue
        lpid=$(cat "$FLEET_DIR/$L.loop" 2>/dev/null || true)
        if [ -n "$lpid" ] && kill -0 "$lpid" 2>/dev/null; then
            if [ "$DRYRUN" = "1" ]; then
                echo "DRYRUN + kill fleet loop pid $lpid ($m)"
            else
                echo "killing fleet loop pid $lpid ($m)"
                kill "$lpid" 2>/dev/null || true
            fi
        fi
        [ "$DRYRUN" = "1" ] || rm -f "$FLEET_DIR/$L.loop"
    done
    for L in $ALL_LABELS; do
        if launchctl print "$GUI/$L" >/dev/null 2>&1; then
            do_run launchctl bootout "$GUI/$L"
        fi
    done
    [ "$DRYRUN" = "1" ] || rm -f /tmp/ds4-v41-server.pid /tmp/ds4-qfn-server.pid \
          /tmp/ds4-v41-serial.pid /tmp/ds4-v41-dspark.pid
    if [ "$DRYRUN" = "1" ]; then return 0; fi

    # Sweep rounds: SIGTERM every server + every lock holder, then wait.
    # Looping (instead of one scan + passive wait) catches a server that
    # raced into existence between the scan and the kills — its loop is
    # already dead, so re-killing is terminal.
    waited=0
    while : ; do
        targets=$( { pgrep -x ds4-server || true; lock_holders; } | sort -un )
        [ -n "$targets" ] || break
        for pid in $targets; do
            echo "SIGTERM pid $pid ($(ps -p "$pid" -o comm= 2>/dev/null || echo unknown)) — engine / lock holder"
            kill "$pid" 2>/dev/null || true
        done
        if [ "$waited" -eq 0 ]; then
            sleep "$MIN_SLEEP"; waited=$MIN_SLEEP
        else
            sleep 2; waited=$((waited + 2))
        fi
        if [ "$waited" -ge "$WAIT_TIMEOUT" ]; then
            echo "ERROR: engine processes still alive after ${waited}s — aborting; lock file NOT deleted, not launching over them" >&2
            echo "  live holders: $(lock_holders | tr '\n' ' ')" >&2
            return 1
        fi
    done
    # Nothing holds it anymore: the operator never deletes it by hand.
    rm -f "$LOCK_FILE"
    return 0
}

# ---- stop ------------------------------------------------------------------
# Halt the fleet regardless of which weights are resident. Stop flags stay
# planted until the next launch of each model clears its own.
stop_all() {
    kill_running || exit 1
    echo "stopped: no ds4-server process remains"
    echo "the fleet stays down until the operator boots a model (no RunAtLoad anymore)"
}

# ---- log -------------------------------------------------------------------
tail_log() {
    LM="${1:-}"
    if [ -n "$LM" ]; then
        case " $MODELS " in *" $LM "*) ;; *) die "unknown model: $LM (known: $MODELS)" ;; esac
        LOGF=$(ls -t "$LOGDIR/$LM.log" "$LOGDIR/$LM"-*.log 2>/dev/null | head -1 || true)
    else
        LOGF=$(ls -t "$LOGDIR"/*.log 2>/dev/null | head -1 || true)
    fi
    [ -n "$LOGF" ] || die "no log found in $LOGDIR for ${LM:-any model}"
    echo "tailing $LOGF — Ctrl-C stops watching (watching does not stop the server)"
    exec tail -n 200 -f "$LOGF"
}

# ---- dispatch (non-load modes) ----------------------------------------------
MODEL="${1:-}"
case "$MODEL" in
status) status_all; exit 0 ;;
log)    shift || true; tail_log "${1:-}" ;;
stop)   stop_all; exit 0 ;;
esac

# Flags after the model name, in any order.
WANT_KV=0
[ -n "$MODEL" ] || { echo "usage: $0 <glm53flash|glm53full|ds41f|muse|qwen38fn> [--kv] [--debug]" >&2
                     echo "       $0 status  |  $0 log [model]  |  $0 stop" >&2; exit 2; }
shift
for _flag in "$@"; do
    case "$_flag" in
    --kv)    WANT_KV=1 ;;
    --debug) DEBUG=1 ;;
    *) echo "usage: $0 <model> [--kv] [--debug]  |  $0 status  |  $0 log [model]  |  $0 stop" >&2; exit 2 ;;
    esac
done

# ---- per-model configuration ----------------------------------------------
# Kept exactly as each model runs today; only the management layer (process
# management, host binding, KV policy) is symmetric across the fleet.
# ENVS: per-model standing envs, space-separated KEY=VALUE (no spaces in
# values), exported into the launch script. Presence-gated knobs: =0 still
# enables where the engine gates on presence; value-parsed knobs take it.
ENVS=""
TOK_OVERRIDE=""
DSPARK=""
MTP_MODEL=""
MTP_TIMING=""
KV_FLAGS=""
case "$MODEL" in
ds41f)
    LABEL=$(label_for ds41f)
    PORT=8005
    BIN="$HOME/ds4-v41/ds4-server"
    WD="$HOME/ds4-v41"
    # 1017 build (341 GB: Engram 188.8 + routed IQ2_XXS 142.4 + dense Q4_K
    # 9.37 GiB), imatrix-baked. The old DeepSeek-V4.1-Flash-Q2.gguf is gone.
    MODEL_GGUF="$HOME/ds4-v41/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
    VISION="$HOME/ds4-v41/gguf/DeepSeek-V4.1-Flash-1017-Vision.gguf"
    CTX=1048576
    BATCHED=""         # DSpark serves only when !batched_mode: ds4_server.c
                       # enters the speculative path behind !batched_mode, so
                       # batching would load the drafter and serve serially.
    PREFILL_CHUNK=""   # audited trial ran WITHOUT --prefill-chunk (engine
                       # graph-selected 2048/4096 ladder); a chunk arm rides
                       # the arm queue, not the standing set.
    MTP=""             # legacy --mtp is not this lane's drafter
    DSPARK=1
    MTP_MODEL="$HOME/ds4-v41/gguf/DeepSeek-V4.1-Flash-DSpark-Q4.gguf"
    ENVS="$ENVS DS4_V41_DSPARK_DECODE_READY=1"
    KVDIR="$HOME/.ds4-v41/server-kv"
    KVSPACE=$KV_SPACE_MB_DS41F
    WANT_KV=1          # standing disk KV: restores a long prefix in ms
                       # instead of re-prefilling it — the cache-miss fix.
    KV_FLAGS="--kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096"
    echo "ds41f standing set: DSpark ON ($MTP_MODEL, admission gate exported, non-batched) + vision ON ($VISION) + disk KV ($KVDIR, ${KVSPACE} MB, cold 393216 / continued 4096)"
    ;;
glm53flash)
    LABEL=$(label_for glm53flash)
    PORT=8000
    BIN="$HOME/ds4/ds4-server"
    WD="$HOME/ds4"
    MODEL_GGUF="$HOME/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf"
    VISION="$HOME/ds4/gguf/GLM-5.3-Flash-Vision-Encoder.gguf"
    CTX=1048576
    BATCHED="3"         # MTP requires non-batched serving (ds4_server.c disables MTP under native session batching)
    PREFILL_CHUNK=""
    MTP=""              # GLM built-in MTP (nextn block in the GGUF): priced 0.4-0.67x on ds4 flash — OFF in serving
    KVDIR="$HOME/.ds4/server-kv"
    KVSPACE=$KV_SPACE_MB_GLM53FLASH
    ;;
glm53full)
    LABEL=$(label_for glm53full)
    PORT=8008
    BIN="$HOME/ds4-glmfull/ds4-server"
    WD="$HOME/ds4-glmfull"
    MODEL_GGUF="$HOME/ds4/gguf/GLM-5.3-BF16src-RoutedIQ2XXS-blk78IQ2XXS-Q4Kattn-shexp-Q4Kslim.gguf"
    VISION=""
    CTX=440000
    BATCHED=""
    PREFILL_CHUNK=""   # GLM full rejects --prefill-chunk (graph-selected chunks)
    MTP=""             # MTP OFF in serving: best measured ~0.55-0.65x post verify-parity fix; measurement rides GLM53FULL_PROFILE=mtp
    KVDIR="$HOME/.ds4-glmfull/server-kv"
    KVSPACE=$KV_SPACE_MB_GLM53FULL
    # Build-B lineage staged 2026-10-05; the served-file era is matched by
    # the banner's binary_sha256 (parser: ~/ds4-glmfull/misc/glm53-ab-parser.py)
    # — with --debug the banner lands in the per-load log.
    # SCRATCH 8192 = one-pass prefill scoring (score_rows=4096; scratch =
    # 4096 x ctx x 4B; the 8192 cap binds only at ctx >= ~524k; trim CTX
    # ~0.089 GiB/1k, NEVER the scratch). Radix + wide scorer and the mode-2
    # two-sweep chain are HARDCODED release paths — their env knobs are inert
    # and are NOT set. Wide-kernel knobs (EXPERT_GEMV_WIDE/ATTN_GATHER_WIDE)
    # and the flush-interval/CACHE_ARENA knobs stay unset (all banked
    # neutral/negative; ledger: misc/ INDEX).
    ENVS="DS4_GLM_INDEXED_PREFILL_SCORE_SCRATCH_MB=8192"
    # Diagnostic arm knobs, caller-visible -> launch script, each off/unset
    # by default (a bare DS4_* on the caller line is dropped — render here):
    if [ "${GLM53FULL_CACHE_ARENA:-0}" = 1 ]; then ENVS="$ENVS DS4_GLM_CACHE_ARENA=1"; fi
    if [ -n "${GLM53FULL_DECODE_FLUSH_INTERVAL:-}" ]; then ENVS="$ENVS DS4_GLM_DECODE_FLUSH_INTERVAL=$GLM53FULL_DECODE_FLUSH_INTERVAL"; fi
    if [ -n "${GLM53FULL_PREFILL_FLUSH_INTERVAL:-}" ]; then ENVS="$ENVS DS4_GLM_PREFILL_FLUSH_INTERVAL=$GLM53FULL_PREFILL_FLUSH_INTERVAL"; fi
    case "${GLM53FULL_PROFILE:-prod}" in
    wb)  ENVS="$ENVS DS4_METAL_DECODE_STAGE_PROFILE=1"
         echo "glm53full profile: WB census — decode stage profile ON (~10 t/s-class decode, bounded log cadence; do NOT abort on either — both are the instrument's designed price)"
         ;;
    mtp) MTP=1 MTP_TIMING=1 MODEL_GGUF="$HOME/ds4/gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf" CTX=400000
         # The MTP measurement vehicle. Reference GGUF with quality heads
         # (head-truth sweep); ctx 400k for comparability with every banked
         # MTP window + the module's KV/head memory inside the guarded rung.
         # Every knob below is caller-visible -> launch script; unset = the
         # banked comparator mode of the prior rounds.
         if [ "${GLM53FULL_MTP_EXACT:-0}" = 1 ]; then ENVS="$ENVS DS4_MTP_EXACT_SAMPLING=1"; fi
         if [ "${GLM53FULL_MTP_DRAFT_SELECT:-0}" = 1 ]; then ENVS="$ENVS DS4_MTP_DRAFT_SELECT=1"; fi
         if [ "${GLM53FULL_MTP_HEAD_FUSE:-0}" = 1 ]; then ENVS="$ENVS DS4_MTP_HEAD_FUSE=1"; fi
         if [ -n "${GLM53FULL_MTP_K:-}" ]; then ENVS="$ENVS DS4_MTP_K=$GLM53FULL_MTP_K"; fi
         if [ -n "${GLM53FULL_MTP_VERIFY_SELECT:-}" ]; then ENVS="$ENVS DS4_MTP_VERIFY_SELECT=$GLM53FULL_MTP_VERIFY_SELECT"; fi
         if [ "${GLM53FULL_MTP_SPEC_DISABLE:-0}" = 1 ]; then ENVS="$ENVS DS4_MTP_SPEC_DISABLE=1"; fi
         if [ "${GLM53FULL_MTP_PROBE:-0}" = 1 ]; then ENVS="$ENVS DS4_GLM_MTP_PROBE=1"; fi
         # Arm-4 fused draft+verify CB (e81c4c1): the W-B side of the
         # re-pricing window (W-A = the same launch WITHOUT these two):
         if [ "${GLM53FULL_MTP_FUSE_CB:-0}" = 1 ]; then ENVS="$ENVS DS4_MTP_FUSE_CB=1"; fi
         if [ "${GLM53FULL_MTP_TOPK2_DEBUG:-0}" = 1 ]; then ENVS="$ENVS DS4_TOPK2_DEBUG=1"; fi
         # Adaptive cost gate: VALUE knob, default ON in the engine — pass
         # GLM53FULL_MTP_ADAPTIVE=0 to pin the plain-off arm (the fused-CB
         # pricing windows ran ADAPTIVE=0). Unset = engine default.
         if [ -n "${GLM53FULL_MTP_ADAPTIVE:-}" ]; then ENVS="$ENVS DS4_MTP_ADAPTIVE=$GLM53FULL_MTP_ADAPTIVE"; fi
         echo "glm53full profile: MTP measurement — reference GGUF, ctx $CTX, scratch 8192 + --mtp --mtp-timing ONLY, ONE load; exact=$(if [ "${GLM53FULL_MTP_EXACT:-0}" = 1 ]; then echo ON; else echo off; fi) draft-select=$(if [ "${GLM53FULL_MTP_DRAFT_SELECT:-0}" = 1 ]; then echo ON; else echo off; fi) head-fuse=$(if [ "${GLM53FULL_MTP_HEAD_FUSE:-0}" = 1 ]; then echo ON; else echo off; fi) K=${GLM53FULL_MTP_K:-1} verify-select=${GLM53FULL_MTP_VERIFY_SELECT:-row} fuse-cb=$(if [ "${GLM53FULL_MTP_FUSE_CB:-0}" = 1 ]; then echo ON; else echo off; fi) topk2-debug=$(if [ "${GLM53FULL_MTP_TOPK2_DEBUG:-0}" = 1 ]; then echo ON; else echo off; fi) adaptive=${GLM53FULL_MTP_ADAPTIVE:-engine-default}"
         ;;
    *)   ;;
    esac
    ;;
muse)
    LABEL=$(label_for muse)
    PORT=8123
    BIN="$HOME/ds4-muse/ds4-server"
    WD="$HOME/ds4-muse"
    # Text-only served build: IQ2_XXS routed + Q4K dense (embed Q8_0, head
    # protected), resident 204.85 GiB; operator envelope 239 GiB. Native ctx
    # 1048576 — NO --ctx clamp (plan at 1M: 204.85 + 33.56 KV = 238.40,
    # fits; boot plan line is the authority). No MTP block in the model.
    MODEL_GGUF="$HOME/ds4-muse/gguf/Muse-Spark-Open-816B-textbase.gguf"
    VISION=""
    CTX=1048576
    BATCHED=""
    PREFILL_CHUNK=""   # engine default 4096 via ds4_prefill_cap_for_prompt
    MTP=""
    # Operator directive (probe-13 era): the clamp stays BELOW ctx — 393216.
    TOK_OVERRIDE=393216
    KVDIR="$HOME/.ds4-muse/server-kv"
    KVSPACE=$KV_SPACE_MB_MUSE
    # Fleet muse serves the production defaults (LEDGER/CB_TIMES/BUSY all
    # off). Instrumented windows are scripts/muse-probe.sh's job; if a fleet
    # window must carry engine envs, use the arm queue (arms-muse) or
    # DS4_FLEET_ENVS — never a bare DS4_* on the caller line.
    ;;
qwen38fn)
    LABEL=$(label_for qwen38fn)
    PORT=8007
    BIN="$HOME/ds4-qfn/ds4-server"
    WD="$HOME/ds4-qfn"
    MODEL_GGUF="$HOME/ds4-qfn/gguf/Qwen3.8-Flash-Next-Q4.gguf"
    VISION=""
    CTX=262144
    BATCHED=4
    PREFILL_CHUNK=8192
    MTP=1
    KVDIR="$HOME/.ds4-qfn/server-kv"
    KVSPACE=$KV_SPACE_MB_QWEN38FN
    ;;
*)
    die "unknown model: '$MODEL' (known: glm53flash glm53full ds41f muse qwen38fn)"
    ;;
esac

[ -n "${TOK_OVERRIDE:-}" ] && TOKENS="$TOK_OVERRIDE"
LAUNCH="$FLEET_DIR/$LABEL.launch.sh"
STOPFLAG="$FLEET_DIR/$LABEL.stop"
LOOPPID="$FLEET_DIR/$LABEL.loop"

# ---- arm queue resolution (before ANY mutation) -----------------------------
ARM_NAME="" ARM_ENVS="" ARM_HOLD=0
if [ -n "${DS4_FLEET_ARM:-}" ]; then
    ARM_NAME="$DS4_FLEET_ARM"; ARM_HOLD=1
elif [ -f "$FLEET_DIR/arm-$MODEL" ]; then
    ARM_NAME=$(head -1 "$FLEET_DIR/arm-$MODEL" | tr -d ' \t\r' || true)
fi
if [ -n "$ARM_NAME" ]; then
    DEFS="$FLEET_DIR/arms-$MODEL"
    [ -f "$DEFS" ] || die "arm '$ARM_NAME' requested but definitions file $DEFS is missing"
    ARM_ENVS=$(awk -F'\t' -v n="$ARM_NAME" '!/^#/ && NF>=2 && $1==n {sub(/^[^\t]*\t/,""); print; exit}' "$DEFS")
    [ -n "$ARM_ENVS" ] || die "arm '$ARM_NAME' not defined in $DEFS (refusing to serve the unpatched default)"
fi
# Render order: standing + profile, then DS4_FLEET_ENVS, then arm (arm wins).
ENVS="$ENVS ${DS4_FLEET_ENVS:-} ${ARM_ENVS:-}"
if [ -n "$ARM_NAME" ]; then
    if [ "$ARM_HOLD" = "1" ]; then
        echo "arm '$ARM_NAME' (DS4_FLEET_ARM override — queue cursor NOT advanced): $ARM_ENVS"
    else
        echo "arm '$ARM_NAME' (queue cursor): $ARM_ENVS"
    fi
fi

arm_advance() {
    # Real load only. Cursor = current name; successor = next def line's name.
    if [ "$ARM_HOLD" = "1" ]; then echo "arm override served — queue cursor unchanged"; return 0; fi
    [ -n "$ARM_NAME" ] || return 0
    DEFS="$FLEET_DIR/arms-$MODEL"
    next=$(awk -F'\t' -v n="$ARM_NAME" '!/^#/ && NF>=2 { if (found) { print $1; exit } if ($1==n) found=1 }' "$DEFS")
    if [ -n "$next" ]; then
        printf '%s\n' "$next" > "$FLEET_DIR/arm-$MODEL"
        echo "queue advanced: '$ARM_NAME' -> '$next' (next $MODEL load serves '$next')"
    else
        rm -f "$FLEET_DIR/arm-$MODEL"
        echo "queue terminal: '$ARM_NAME' was the last arm — cursor removed, next $MODEL load serves standing production"
    fi
}

# ---- launch script generation ----------------------------------------------
build_cmd() {
    CMD="\"$BIN\" --metal -m \"$MODEL_GGUF\""
    [ -n "$VISION" ] && CMD="$CMD --vision \"$VISION\""
    CMD="$CMD --ctx $CTX --tokens $TOKENS --threads $THREADS --power $POWER --host $HOST --port $PORT"
    [ -n "$BATCHED" ] && CMD="$CMD --batched-session $BATCHED"
    [ -n "$PREFILL_CHUNK" ] && CMD="$CMD --prefill-chunk $PREFILL_CHUNK"
    [ -n "$MTP" ] && CMD="$CMD --mtp"
    [ -n "$MTP_TIMING" ] && CMD="$CMD --mtp-timing"
    [ -n "$DSPARK" ] && CMD="$CMD --dspark"
    [ -n "$MTP_MODEL" ] && CMD="$CMD --mtp-model \"$MTP_MODEL\""
    if [ "$WANT_KV" = "1" ]; then
        mkdir -p "$KVDIR"
        CMD="$CMD --kv-disk-dir \"$KVDIR\" --kv-disk-space-mb $KVSPACE"
        [ -n "$KV_FLAGS" ] && CMD="$CMD $KV_FLAGS"
    fi
    return 0
}

write_launch_script() {
    mkdir -p "$FLEET_DIR" "$LOGDIR"
    LOGFILE="$LOGDIR/$MODEL.log"
    RUNLOG=""
    [ "$DEBUG" = "1" ] && RUNLOG="$LOGDIR/$MODEL-$(date +%Y%m%d-%H%M%S).log"
    TEE="tee -a \"$LOGFILE\""
    [ -n "$RUNLOG" ] && TEE="tee -a \"$LOGFILE\" \"$RUNLOG\""

    {
        echo "#!/bin/sh"
        echo "# generated by fleet-server.sh $(date '+%F %T') — model $MODEL ($LABEL) — do not edit; re-generate via the switcher"
        echo "# cmd: $CMD"
        echo "# envs: $ENVS"
        echo "# logging: always-on per-model log $LOGFILE (via tee)$( [ "$DEBUG" = "1" ] && echo " + per-load debug log $RUNLOG")"
        echo "cd \"$WD\" || exit 1"
        echo "rm -f \"$STOPFLAG\""
        # Console -> the launching terminal pane. The subshell probe keeps a
        # tty-less launch (cron, ssh -T) from dying on a failed exec redirect.
        echo 'if ( exec >/dev/tty ) 2>/dev/null; then exec >/dev/tty 2>&1; fi'
        echo "RUNLOG=\"$RUNLOG\""
        echo "LOGFILE=\"$LOGFILE\""
        echo 'say() { echo "$*"; [ -n "$LOGFILE" ] && echo "$*" >> "$LOGFILE"; [ -n "$RUNLOG" ] && echo "$*" >> "$RUNLOG"; }'
        echo "say \"=== fleet launch: $MODEL ($LABEL) port $PORT $(date '+%F %T') ===\""
        echo "export HOME=\"$HOME\""
        echo "export DS4_LOCK_FILE=\"$LOCK_FILE\""
        for _kv in $ENVS; do
            [ -n "$_kv" ] || continue
            _k="${_kv%%=*}" ; _v="${_kv#*=}"
            echo "export $_k=\"$_v\""
        done
        echo 'ECFILE=$(mktemp -t ds4fleet-ec)'
        echo "while : ; do"
        # Re-check the stop flag at the TOP of every iteration: a loop that
        # was sitting in the throttle sleep must NOT boot a new server
        # after a halt planted the flag (the stale-lock race).
        echo '    [ -f "'"$STOPFLAG"'" ] && { say "=== fleet: '"$MODEL"' stop flag set before launch — not launching ==="; break; }'
        echo "        { $CMD; echo \$? > \"\$ECFILE\"; } 2>&1 | $TEE"
        echo '        ec=$(cat "$ECFILE")'
        echo '    [ "$ec" -eq 0 ] && { say "=== fleet: '"$MODEL"' clean exit — not relaunching ==="; break; }'
        echo '    [ -f "'"$STOPFLAG"'" ] && { say "=== fleet: '"$MODEL"' exited $ec with stop flag set — not relaunching ==="; break; }'
        echo '    say "=== fleet: '"$MODEL"' exited $ec — relaunching in '"$THROTTLE"' s (KeepAlive semantics) ==="'
        echo "    sleep $THROTTLE"
        echo 'done'
        echo 'rm -f "$ECFILE"'
    } > "$LAUNCH"
    chmod +x "$LAUNCH"
    sh -n "$LAUNCH" || die "generated launch script $LAUNCH fails sh -n"
}

# ---- pre-flight gates (before ANY mutation) ---------------------------------
[ -x "$BIN" ] || die "$BIN not executable — build the worktree (make ds4-server)"
[ -f "$MODEL_GGUF" ] || die "model file missing: $MODEL_GGUF"
[ -z "$VISION" ] || [ -f "$VISION" ] || die "vision file missing: $VISION"
[ -z "$MTP_MODEL" ] || [ -f "$MTP_MODEL" ] || die "DSpark support file missing: $MTP_MODEL"
host_bindable || die "bind address $HOST is not assigned to any local interface — refusing to launch (a wrong bind address passes every later gate and costs a full weight mapping per crash-loop round)"

build_cmd
# argv gate: flag parsing happens before any model load and before the
# instance lock — a bad flag exits nonzero without touching the resident
# server. Run it against the real binary BEFORE halting anything.
( eval "$CMD --help" ) >/dev/null 2>&1 \
    || die "argv rejected by $BIN (see: eval \"$CMD --help\")"

write_launch_script
KVSTATE="no ssd kv cache"
[ "$WANT_KV" = "1" ] && KVSTATE="ssd kv cache: $KVDIR (budget ${KVSPACE} MB)"
LOGSTATE="per-model log $LOGDIR/$MODEL.log (console streams to the launching terminal too)"
[ "$DEBUG" = "1" ] && LOGSTATE="$LOGSTATE + per-load debug log: $RUNLOG"

kill_running || exit 1

if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN: launch script written at $LAUNCH ($KVSTATE, $LOGSTATE)"
    echo "DRYRUN: would run: nohup /bin/sh $LAUNCH >/dev/null 2>&1 &"
    echo "DRYRUN: arm cursor NOT advanced"
    exit 0
fi

# Final lock gate (belt over the halt's own rm): if anything still holds
# the instance lock, the load would die inside the engine — say exactly
# who instead.
if [ -e "$LOCK_FILE" ]; then
    _hold=$(lock_holders | tr '\n' ' ')
    if [ -n "$_hold" ]; then
        die "instance lock $LOCK_FILE still held by pid(s): $_hold — refusing to load"
    fi
    rm -f "$LOCK_FILE"
fi

# Detached background load: the switcher's job ends here. No terminal, no
# session, no window — just the model coming up. The loop attaches its own
# console output to THIS pane via exec >/dev/tty inside the launch script.
nohup /bin/sh "$LAUNCH" >/dev/null 2>&1 &
LOOP_PID=$!
echo "$LOOP_PID" > "$LOOPPID"
kill -0 "$LOOP_PID" 2>/dev/null \
    && echo "launched $MODEL as detached loop pid $LOOP_PID (port $PORT, $KVSTATE, $LOGSTATE)" \
    || die "loop pid $LOOP_PID not alive after launch"
arm_advance
echo "weights take 45-90s to map"
echo "to halt by hand: plant $STOPFLAG first, then SIGTERM the server — the loop otherwise relaunches in ${THROTTLE}s"
