#!/bin/bash
# FastAPI GenZ Creator Search API Startup Script

echo "🚀 Starting FastAPI GenZ Creator Search API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if requirements have changed or first run
DEPS_SENTINEL="venv/.requirements-installed"
if [ ! -f "${DEPS_SENTINEL}" ] || [ requirements.txt -nt "${DEPS_SENTINEL}" ]; then
    echo "📥 Installing/updating dependencies..."
    pip install -r requirements.txt
    touch "${DEPS_SENTINEL}"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    exit 1
fi

# Load environment variables (including PORT) from .env
set -a
source .env
set +a

# Resolve LanceDB location (prefers DIME-AI-DB/data/lancedb)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
CANDIDATES=()

if [ -n "${DIME_AI_DB_ROOT}" ]; then
    CANDIDATES+=("${DIME_AI_DB_ROOT}")
fi

CANDIDATES+=(
    "${WORKSPACE_ROOT}/DIME-AI-DB"
    "${PROJECT_ROOT}/DIME-AI-DB"
)

function pick_db_path() {
    for base in "${CANDIDATES[@]}"; do
        if [ -d "${base}/data/lancedb" ]; then
            echo "${base}/data/lancedb"
            return 0
        fi
    done
    for base in "${CANDIDATES[@]}"; do
        if [ -d "${base}/influencers_vectordb" ]; then
            echo "${base}/influencers_vectordb"
            return 0
        fi
    done
    return 1
}

if resolved_path="$(pick_db_path)"; then
    export DB_PATH="${DB_PATH:-$resolved_path}"
    echo "📂 Using LanceDB at: ${DB_PATH}"
else
    echo "⚠️  Could not auto-detect LanceDB directory."
    echo "   Set DIME_AI_DB_ROOT or DB_PATH to point at DIME-AI-DB/data/lancedb."
fi

# Start the FastAPI server
echo "🌐 Starting FastAPI server..."
APP_PORT="${PORT:-7001}"

echo "📡 API will be available at: http://localhost:${APP_PORT}"
echo "📖 API documentation at: http://localhost:${APP_PORT}/docs"
echo "📚 Alternative docs at: http://localhost:${APP_PORT}/redoc"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}" --reload
