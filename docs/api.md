# GenZ Creator Search API Documentation

A FastAPI-based REST API for searching and managing GenZ creators/influencers with advanced matching algorithms and real-time image refresh capabilities.

## Base URLs

- Local development: `http://localhost:7001`
- Production (pen.optimat.us): `http://pen.optimat.us:7001`

## Interactive Documentation

- **Swagger UI**: `http://localhost:7001/docs`
- **ReDoc**: `http://localhost:7001/redoc`
- **Production (pen.optimat.us)**: `http://pen.optimat.us:7001/docs`

## API Root

All JSON endpoints are served beneath the `/search` path (for example: `POST /search/`, `POST /search/similar`). Image refresh/proxy functionality has moved to the companion `DIME-AI-BD` service (`/brightdata/images/...`).

## Authentication

Currently, no authentication is required. In production, consider implementing API key authentication.

## API Endpoints

### Health Check

#### GET `/health`
Check API health status

**Response:**
```json
{
  "status": "healthy",
  "database_available": true,
  "timestamp": "2024-07-15T12:34:56+00:00"
}
```

- `status`: `"healthy"` when all dependencies are available, `"degraded"` otherwise
- `database_available`: `true` when the LanceDB instance is reachable
- `timestamp`: ISO 8601 timestamp for when the check was processed

---

## Search Endpoints

### Search Creators

#### POST `/search/`
Search for creators based on business description or query

**Request Body:**
```json
{
  "query": "sustainable fashion brand targeting millennials",
  "is_business_description": false,
  "method": "hybrid",
  "limit": 20,
  "min_followers": 1000,
  "max_followers": 10000000,
  "min_engagement": 0.0,
  "location": "New York",
  "category": "Fashion",
  "keywords": ["sustainable", "eco-friendly"],
  "weights": {
    "business_alignment": 0.30,
    "genz_appeal": 0.25,
    "authenticity": 0.20,
    "engagement": 0.15,
    "campaign_value": 0.10
  }
}
```

**Parameters:**
- `query` (string, required): Business description or search query
- `is_business_description` (boolean): Whether query is a business description
- `method` (string): Search method - "vector", "text", or "hybrid" (default: "hybrid")
- `limit` (integer): Maximum results (1-100, default: 20)
- `min_followers` (integer): Minimum follower count (default: 1000)
- `max_followers` (integer): Maximum follower count (default: 10000000)
- `min_engagement` (float): Minimum engagement rate (default: 0.0)
- `location` (string, optional): Location filter
- `category` (string, optional): Business category filter
- `keywords` (array, optional): Additional keywords
- `weights` (object, optional): Custom scoring weights

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": 123456,
      "account": "sustainablestyle_sarah",
      "profile_name": "Sarah Johnson",
      "followers": 15000,
      "followers_formatted": "15.0K",
      "avg_engagement": 0.045,
      "business_category_name": "Fashion",
      "business_address": "New York, NY",
      "biography": "Sustainable fashion advocate | Eco-friendly lifestyle tips",
      "profile_image_link": "https://...",
      "posts": [...],
      "score": 0.92,
      "engagement_score": 0.87,
      "relevance_score": 0.95,
      "genz_appeal_score": 0.88,
      "authenticity_score": 0.91,
      "campaign_value_score": 0.85,
      "category_relevance_score": 0.93,
      "business_alignment_score": 0.89,
      "is_personal_creator": true
    }
  ],
  "count": 15,
  "query": "sustainable fashion brand targeting millennials",
  "method": "hybrid"
}
```

### Find Similar Creators

#### POST `/search/similar`
Find creators similar to a reference account

**Request Body:**
```json
{
  "account": "reference_username",
  "limit": 10,
  "min_followers": 1000
}
```

**Parameters:**
- `account` (string, required): Reference account username
- `limit` (integer): Maximum results (1-50, default: 10)
- `min_followers` (integer): Minimum follower count (default: 1000)

### Search by Category

#### POST `/search/category`
Search creators by business category

**Request Body:**
```json
{
  "category": "Fashion",
  "location": "Los Angeles",
  "limit": 15,
  "min_followers": 5000
}
```

**Parameters:**
- `category` (string, required): Category to search for
- `location` (string, optional): Location filter
- `limit` (integer): Maximum results (1-50, default: 15)
- `min_followers` (integer): Minimum follower count (default: 5000)

---

## Image Management Endpoints

Image refresh/proxy functionality now lives in the `DIME-AI-BD` service. Use its `/brightdata/images/...` routes for batch refreshes, job polling, SSE streams, and CDN proxies.

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed error information"
}
```

### Common HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `408`: Request Timeout
- `500`: Internal Server Error
- `501`: Not Implemented
- `503`: Service Unavailable

---

## Data Models

### SearchResult
```json
{
  "id": "integer",
  "account": "string",
  "profile_name": "string",
  "followers": "integer",
  "followers_formatted": "string",
  "avg_engagement": "float",
  "business_category_name": "string",
  "business_address": "string",
  "biography": "string",
  "profile_image_link": "string",
  "posts": "array",
  "score": "float (0.0-1.0)",
  "engagement_score": "float (0.0-1.0)",
  "relevance_score": "float (0.0-1.0)",
  "genz_appeal_score": "float (0.0-1.0)",
  "authenticity_score": "float (0.0-1.0)",
  "campaign_value_score": "float (0.0-1.0)",
  "category_relevance_score": "float (0.0-1.0)",
  "business_alignment_score": "float (0.0-1.0)",
  "is_personal_creator": "boolean"
}
```

### Custom Weights
```json
{
  "business_alignment": "float (0.0-1.0)",
  "genz_appeal": "float (0.0-1.0)",
  "authenticity": "float (0.0-1.0)",
  "engagement": "float (0.0-1.0)",
  "campaign_value": "float (0.0-1.0)"
}
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
DEBUG=false

# Database (defaults to ../DIME-AI-DB/data/lancedb when available)
DB_PATH=/path/to/DIME-AI-DB/data/lancedb

# Image Refresh Service (DIME-AI-BD)
BRIGHTDATA_SERVICE_URL=http://localhost:7100/brightdata/images
BRIGHTDATA_JOB_TIMEOUT=600
BRIGHTDATA_JOB_POLL_INTERVAL=5

# CORS (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

Make sure the companion `DIME-AI-BD` service is running (including Redis + RQ worker) at the URL specified by `BRIGHTDATA_SERVICE_URL`.

---

## Rate Limiting

Currently no rate limiting is implemented. Consider adding rate limiting for production use:

- Search endpoints: 60 requests/minute
- Image refresh: 10 requests/minute
- Proxy images: 120 requests/minute

---

## Best Practices

### Search Optimization
1. Use specific business descriptions for better matching
2. Set appropriate follower ranges for your campaign
3. Use custom weights to prioritize important factors
4. Include relevant keywords for better filtering

### Image Management
1. Refresh images in batches (max 50 users)
2. Monitor service status before making requests
3. Use the proxy endpoint for client-side image display

### Performance
1. Use pagination with reasonable limits
2. Cache results on the client side when possible
3. Use specific filters to reduce result sets

---

## Migration from Flask

If migrating from the Flask version:

1. Update base URL from `localhost:5001` to `localhost:7001`
2. Update request/response format to match new schemas
3. Replace `/search` with `/search/`
4. Replace `/similar` with `/search/similar`
5. Replace `/category` with `/search/category`
6. Update legacy `/api/refresh-images` or `/api/proxy-image` calls to point at `/brightdata/images/...` on the BrightData service.

---

## Support

For issues or questions:
1. Check the interactive documentation at `/docs`
2. Review error messages in responses
3. Ensure all required environment variables are set
4. Verify database availability with the health check endpoint
