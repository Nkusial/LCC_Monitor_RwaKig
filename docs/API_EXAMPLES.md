# API Examples

Base URL:

```text
http://127.0.0.1:8000
```

## Health

```http
GET /health
```

Example:

```json
{"status": "ok"}
```

## API Landing

```http
GET /
```

Example:

```json
{
  "name": "Confidence-Aware Land-Cover Change Monitoring API",
  "status": "ok",
  "docs": "/docs",
  "endpoints": ["/health", "/aoi", "/changes", "/changes/summary", "/changes/geojson"]
}
```

## Change Summary

```http
GET /changes/summary
```

Example:

```json
{
  "total": 2076,
  "mean_confidence": 0.877,
  "total_area_m2": 60506400.0,
  "by_monitored_land_cover": {
    "bare_sparse": 698,
    "vegetation": 607,
    "built_up": 291,
    "mixed": 281,
    "water_moisture": 199
  },
  "by_reliability": {
    "medium": 1119,
    "high": 957
  }
}
```

## Change GeoJSON

```http
GET /changes/geojson?limit=1000&min_confidence=0.75&monitored_land_cover=vegetation
```

Each feature includes:

```text
change_type
monitored_land_cover
transition
before_state
after_state
confidence
area_m2
change_magnitude
before_date
after_date
```

## Change Records

```http
GET /changes?limit=500
```

Example item:

```json
{
  "id": "39ffdd72-0bc4-4549-8eab-65a01048e8b2",
  "change_type": "radar_change",
  "confidence": 0.758,
  "area_m2": 11400.0,
  "monitored_land_cover": "vegetation",
  "transition": "vegetation_to_mixed",
  "before_date": "2025-01-06",
  "after_date": "2025-02-20",
  "reliability": "medium"
}
```
