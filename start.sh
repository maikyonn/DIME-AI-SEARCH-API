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

# Install dependencies if requirements have changed
if [ requirements.txt -nt venv/pyvenv.cfg ]; then
    echo "📥 Installing/updating dependencies..."
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    exit 1
fi

# Resolve combined LanceDB vector path relative to this repo
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VDB_PATH="${PROJECT_ROOT}/../DIME-AI-DB/data/combined/influencers_vectordb"

if [ -d "$VDB_PATH" ]; then
    export DB_PATH="$VDB_PATH"
    echo "📂 Using combined vector database at: $DB_PATH"
else
    echo "⚠️  Combined vector database not found at $VDB_PATH"
    echo "   Falling back to path from configuration (.env or defaults)."
fi

# Start the FastAPI server
echo "🌐 Starting FastAPI server..."
echo "📡 API will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "📚 Alternative docs at: http://localhost:8000/redoc"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
