#!/bin/sh
# Control for the GLM 5.3 Flash workhorse server (launchd-managed).
#
# The plist ~/Library/LaunchAgents/com.ds4-glm53-flash.server.plist runs
# /Users/max/ds4/ds4-server with KeepAlive on port 8000. This wrapper is a
# thin launchctl front end so all three models are operated the same way:
#   ~/ds4/run-server.sh      GLM 5.3 Flash (this one), launchd, port 8000
#   ~/ds4-v41/run-server.sh  DeepSeek V4.1 Flash Q2, nohup, port 8005
#   ~/ds4-qfn/run-server.sh  Qwen3.8 Flash Next Q4, nohup, port 8007
# The instance lock allows ONE resident huge-model process: stop the running
# one before starting another.
#
# KV_SPACE_MB is not passed to the server by this wrapper (the plist carries
# the server flags; the live server currently runs without a disk KV store --
# Hermes owns client-side KV policy). It states the GLM store's budget for
# purge-kv-caches.sh envelope accounting, which reads it from this file while
# the legacy ~/.ds4/server-kv store drains through the 72h purge.
#
# Logs: the plist redirects to ~/Library/Logs/ds4-glm-5.3-server/{out,err}.log,
# rotated fleet-wide by ~/ds4/rotate-logs.sh (launchd com.ds4.log-rotation,
# daily 03:00, 7 day retention, copytruncate so no restart is needed).
#
# Usage: ./run-server.sh [start|stop|restart|status]
set -eu
LABEL=com.ds4-glm5.3-flash.server
PLIST="$HOME/Library/LaunchAgents/com.ds4-glm53-flash.server.plist"
KV_SPACE_MB=1048576   # read by purge-kv-caches.sh, see comment above

loaded()  { launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; }
running() { pgrep -x ds4-server >/dev/null 2>&1; }

case "${1:-status}" in
start)
    if running; then
        echo "a ds4-server is already running (instance lock):"
        pgrep -lx ds4-server
        exit 1
    fi
    if loaded; then
        launchctl kickstart "gui/$(id -u)/$LABEL"
    else
        launchctl bootstrap "gui/$(id -u)" "$PLIST"
    fi
    echo "GLM 5.3 Flash server start requested (label $LABEL)"
    ;;
stop)
    if loaded; then
        launchctl bootout "gui/$(id -u)/$LABEL"
        echo "GLM server stopped (unloaded $LABEL)"
    else
        echo "GLM server not loaded"
    fi
    ;;
restart)
    launchctl kickstart -k "gui/$(id -u)/$LABEL"
    echo "GLM server restarted"
    ;;
status)
    if loaded; then
        launchctl print "gui/$(id -u)/$LABEL" | grep -E 'state =|pid =' | head -2
    else
        echo "not loaded"
    fi
    if running; then pgrep -lx ds4-server; else echo "no ds4-server process"; fi
    ;;
*)
    echo "usage: $0 [start|stop|restart|status]"
    exit 2
    ;;
esac
