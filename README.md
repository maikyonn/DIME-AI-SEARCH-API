# GenZ Creator Search FastAPI

A modern FastAPI implementation of the GenZ Creator Search system, providing RESTful APIs for searching and managing influencers/creators with advanced matching algorithms and real-time image refresh capabilities.

## 🚀 Features

- **Advanced Search**: Hybrid vector + text search with business description matching
- **Similarity Search**: Find creators similar to reference accounts
- **Category Filtering**: Search by business categories with location filters
- **Image Refresh**: Real-time profile image updates via Bright Data integration
- **Custom Scoring**: Configurable weights for business alignment, authenticity, engagement, etc.
- **Auto Documentation**: Interactive Swagger UI and ReDoc documentation
- **Type Safety**: Full Pydantic validation and type hints
- **Async Support**: Asynchronous operations for better performance

## 📁 Project Structure

```
fastapi_backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration settings
│   ├── dependencies.py         # Shared dependencies
│   │
│   ├── api/v1/                # API endpoints
│   │   ├── router.py          # Main API router
│   │   ├── search.py          # Search endpoints
│   │   ├── creators.py        # Creator endpoints
│   │   └── images.py          # Image management endpoints
│   │
│   ├── core/                  # Core business logic
│   │   └── search_engine.py   # Search engine wrapper
│   │
│   ├── models/                # Pydantic models
│   │   ├── search.py          # Search request/response models
│   │   └── creator.py         # Creator data models
│   │
│   └── services/              # External service integrations
│       └── image_refresh.py   # Image refresh service wrapper
│
├── docs/
│   └── api.md                 # Comprehensive API documentation
│
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Access to the existing LanceDB database
- (Optional) Bright Data API token for image refresh functionality

### Installation

1. **Clone or copy the fastapi_backend folder**

2. **Create virtual environment**
   ```bash
   cd fastapi_backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Ensure database access**
   - The API expects the LanceDB database at `../snap_data_lancedb` relative to the project root
   - Or set `DB_PATH` in your `.env` file to point to the correct location

### Configuration

Edit `.env` file with your settings:

```env
# Set to true for development
DEBUG=true

# Path to LanceDB database (optional - defaults to ../snap_data_lancedb)
DB_PATH=/path/to/your/snap_data_lancedb

# Bright Data API token for image refresh (optional)
BRIGHTDATA_API_TOKEN=your_token_here

# CORS origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## 🚀 Running the Application

### Development Server

```bash
# Start with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Server

```bash
# Start production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Alternative: Direct Python

```bash
python -m app.main
```

## 📖 API Documentation

### Interactive Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Comprehensive Documentation

See [docs/api.md](docs/api.md) for detailed API documentation including:
- All endpoints with examples
- Request/response schemas
- Error handling
- Configuration options
- Migration guide from Flask

## 🔧 API Endpoints Overview

### Health & Status
- `GET /health` - Health check
- `GET /` - API information

### Search
- `POST /api/v1/search/` - Main creator search
- `POST /api/v1/search/similar` - Find similar creators
- `POST /api/v1/search/category` - Category-based search

### Image Management
- `POST /api/v1/images/refresh` - Refresh profile images
- `POST /api/v1/images/refresh/search-results` - Refresh from search results
- `GET /api/v1/images/refresh/status` - Service status
- `GET /api/v1/images/proxy` - Image proxy (CORS bypass)

## 🔍 Usage Examples

### Basic Search
```python
import requests

response = requests.post('http://localhost:8000/api/v1/search/', json={
    "query": "sustainable fashion brand targeting Gen Z",
    "limit": 10,
    "min_followers": 5000
})

results = response.json()
```

### Custom Weighted Search
```python
response = requests.post('http://localhost:8000/api/v1/search/', json={
    "query": "tech startup focusing on mobile apps",
    "method": "hybrid",
    "limit": 20,
    "weights": {
        "business_alignment": 0.35,
        "genz_appeal": 0.25,
        "authenticity": 0.20,
        "engagement": 0.15,
        "campaign_value": 0.05
    }
})
```

### Find Similar Creators
```python
response = requests.post('http://localhost:8000/api/v1/search/similar', json={
    "account": "reference_username",
    "limit": 15,
    "min_followers": 10000
})
```

## 🔄 Migration from Flask

If you're migrating from the Flask version:

1. **URL Changes**:
   - Base URL: `localhost:5001` → `localhost:8000`
   - Add `/api/v1` prefix to endpoints
   - `/search` → `/api/v1/search/`
   - `/similar` → `/api/v1/search/similar`
   - `/category` → `/api/v1/search/category`

2. **Request Format**: Now uses Pydantic models for validation
3. **Response Format**: Standardized with success/error fields
4. **Error Handling**: HTTP status codes with detailed error messages

## ⚡ Performance Features

- **Async Operations**: Non-blocking I/O for better concurrency
- **Dependency Injection**: Efficient resource management
- **Pydantic Validation**: Fast request/response validation
- **Auto Documentation**: No performance overhead for docs
- **Connection Pooling**: Efficient database connections

## 🛡️ Security Considerations

- **Input Validation**: All inputs validated with Pydantic
- **CORS Configuration**: Configurable cross-origin policies
- **Error Handling**: Safe error messages without sensitive data
- **Rate Limiting**: Consider adding for production use

## 📝 Development

### Code Structure

- **Clean Architecture**: Separation of concerns with layers
- **Type Safety**: Full type hints and Pydantic models
- **Error Handling**: Comprehensive exception handling
- **Documentation**: Auto-generated and manual documentation

### Adding New Features

1. Create Pydantic models in `app/models/`
2. Implement business logic in `app/core/` or `app/services/`
3. Add endpoints in appropriate `app/api/v1/` modules
4. Update documentation

## 🔧 Troubleshooting

### Common Issues

1. **Database not found**: Ensure `DB_PATH` points to valid LanceDB directory
2. **Import errors**: Check Python path and dependencies
3. **Port conflicts**: Change port with `--port` flag
4. **CORS issues**: Update `ALLOWED_ORIGINS` in configuration

### Debugging

Enable debug mode in `.env`:
```env
DEBUG=true
```

This provides detailed error messages and stack traces.

## 📊 Monitoring

Consider adding these for production:

- **Logging**: Structured logging with appropriate levels
- **Metrics**: Request/response metrics and performance monitoring
- **Health Checks**: Extended health checks for dependencies
- **Rate Limiting**: API rate limiting and abuse prevention

## 🤝 Contributing

1. Follow existing code structure and patterns
2. Add proper type hints and documentation
3. Update API documentation for new endpoints
4. Test with both development and production configurations

## 📄 License

Same license as the original project.