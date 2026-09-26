#!/usr/bin/env bash
# Launch ThetaData Terminal v3 in background and verify it's ready.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAR="$SCRIPT_DIR/ThetaTerminalv3.jar"
CONFIG="$SCRIPT_DIR/config.toml"
CREDS="$SCRIPT_DIR/creds.txt"
LOG_DIR="$HOME/thetadata/logs"
LOG_FILE="$LOG_DIR/terminal.log"
BASE_URL="http://127.0.0.1:25503"

# ─── Prefer Homebrew JDK 21 — set JAVA_HOME so sub-processes inherit it ──────
BREW_JDK_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if [[ -d "$BREW_JDK_HOME" ]]; then
    export JAVA_HOME="$BREW_JDK_HOME"
    export PATH="$JAVA_HOME/bin:$PATH"
    JAVA="$JAVA_HOME/bin/java"
elif command -v java &>/dev/null; then
    JAVA="java"
else
    echo "ERROR: java not found. Install JDK 21+:"
    echo "  macOS:  brew install openjdk@21"
    echo "  Linux:  sudo apt install openjdk-21-jre"
    exit 1
fi
echo "Using Java: $("$JAVA" -version 2>&1 | head -1)"

if [[ ! -f "$JAR" ]]; then
    echo "ERROR: JAR not found at $JAR"
    exit 1
fi

if [[ ! -f "$CREDS" ]]; then
    echo "ERROR: creds.txt not found at $CREDS"
    echo "  Expected format: line 1 = email, line 2 = password"
    exit 1
fi

# Validate creds file is non-empty (terminal validates content itself)
if [[ ! -s "$CREDS" ]]; then
    echo "ERROR: creds.txt is empty — needs email on line 1, password on line 2"
    exit 1
fi

# ─── Kill existing terminal if running ───────────────────────────────────────
if pgrep -f "ThetaTerminalv3.jar" >/dev/null 2>&1; then
    echo "Stopping existing terminal..."
    pkill -f "ThetaTerminalv3.jar" || true
    sleep 2
fi

# ─── Launch ──────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
echo "Starting ThetaData Terminal v3..."
"$JAVA" -jar "$JAR" --creds-file "$CREDS" > "$LOG_FILE" 2>&1 &
PID=$!
echo "PID: $PID"

# ─── Wait for readiness ─────────────────────────────────────────────────────
echo "Waiting for terminal to initialize..."
MAX_WAIT=60
WAITED=0
while (( WAITED < MAX_WAIT )); do
    if curl -s --max-time 3 "$BASE_URL/v3/index/list/symbols" >/dev/null 2>&1; then
        echo "Terminal ready (${WAITED}s)"
        curl -s "$BASE_URL/v3/index/list/symbols" | head -c 200
        echo ""
        exit 0
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    printf "."
done
echo ""
echo "ERROR: Terminal did not become ready after ${MAX_WAIT}s"
echo "Check logs: $LOG_FILE"
exit 1
