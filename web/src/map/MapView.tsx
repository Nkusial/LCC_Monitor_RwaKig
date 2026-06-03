// MapLibre rendering layer for AOI, satellite overlays, and change features.
import { useEffect, useRef } from 'react'
import type { FeatureCollection, Geometry, Point } from 'geojson'
import maplibregl, {
  type ExpressionSpecification,
  type GeoJSONSource,
  type Map,
} from 'maplibre-gl'
import { demoPath, loadAoi, type RasterLayer } from '../api'
import {
  changeWindowLabel,
  fallbackAoi,
  labelForChangeType,
  labelForClass,
  labelForReliability,
  landCoverColors,
  reliabilityColors,
  transitionParts,
} from './layers'

type MapViewProps = {
  changes: FeatureCollection<Geometry>
  selectedLandCover: string
  minConfidence: number
  activeRasterLayer: RasterLayer | null
  vectorColorMode: 'monitored' | 'baseline' | 'after' | 'magnitude' | 'reliability'
  selfSupervisedReview: FeatureCollection<Geometry>
  showSelfSupervisedReview: boolean
}

function collectCoordinatePairs(value: unknown, pairs: number[][]) {
  // GeoJSON polygon nesting varies by geometry type; recursively collect all
  // coordinate pairs so a lightweight centroid can be derived for point symbols.
  if (!Array.isArray(value)) {
    return
  }

  if (
    value.length >= 2 &&
    typeof value[0] === 'number' &&
    typeof value[1] === 'number'
  ) {
    pairs.push([value[0], value[1]])
    return
  }

  value.forEach((item) => collectCoordinatePairs(item, pairs))
}

function boundsFromFeatureCollection(
  collection: FeatureCollection<Geometry>,
): [[number, number], [number, number]] | null {
  // Opening the map from AOI bounds prevents reviewers from landing on a
  // generic basemap view and having to search for the Kigali monitoring area.
  const pairs: number[][] = []
  collection.features.forEach((feature) => {
    if (feature.geometry && 'coordinates' in feature.geometry) {
      collectCoordinatePairs(feature.geometry.coordinates, pairs)
    }
  })

  if (pairs.length === 0) {
    return null
  }

  const lngs = pairs.map(([lng]) => lng)
  const lats = pairs.map(([, lat]) => lat)
  return [
    [Math.min(...lngs), Math.min(...lats)],
    [Math.max(...lngs), Math.max(...lats)],
  ]
}

function fitMapToAoi(map: Map, bounds: [[number, number], [number, number]]) {
  // MapLibre can compute a poor first camera if the canvas is still settling
  // after the sidebar/layout has rendered. Resizing first and capping zoom keeps
  // the complete 20 x 20 km AOI visible when the hosted link opens.
  map.resize()
  map.fitBounds(bounds, {
    padding: { top: 56, bottom: 56, left: 56, right: 56 },
    maxZoom: 12.2,
    duration: 0,
  })
}

function scheduleInitialAoiFit(
  map: Map,
  bounds: [[number, number], [number, number]],
) {
  fitMapToAoi(map, bounds)
  window.requestAnimationFrame(() => fitMapToAoi(map, bounds))
  window.setTimeout(() => fitMapToAoi(map, bounds), 350)
}

function changePointsFromPolygons(
  changes: FeatureCollection<Geometry>,
): FeatureCollection<Point> {
  // Points make dense change features legible at portfolio screenshot scale,
  // while polygons still carry the detailed geometry and popup attributes.
  return {
    type: 'FeatureCollection',
    features: changes.features.flatMap((feature) => {
      const pairs: number[][] = []
      if (feature.geometry && 'coordinates' in feature.geometry) {
        collectCoordinatePairs(feature.geometry.coordinates, pairs)
      }

      if (pairs.length === 0) {
        return []
      }

      const lngs = pairs.map(([lng]) => lng)
      const lats = pairs.map(([, lat]) => lat)
      const center: [number, number] = [
        (Math.min(...lngs) + Math.max(...lngs)) / 2,
        (Math.min(...lats) + Math.max(...lats)) / 2,
      ]

      return [
        {
          type: 'Feature' as const,
          properties: feature.properties,
          geometry: {
            type: 'Point' as const,
            coordinates: center,
          },
        },
      ]
    }),
  }
}

const monitoredGroupColorExpression: ExpressionSpecification = [
  // Color by the land-cover group under monitoring.
  'match',
  ['get', 'monitored_land_cover'],
  'vegetation',
  landCoverColors.vegetation,
  'built_up',
  landCoverColors.built_up,
  'water_moisture',
  landCoverColors.water_moisture,
  'bare_sparse',
  landCoverColors.bare_sparse,
  'mixed',
  landCoverColors.mixed,
  landCoverColors.unknown,
]

const stateColorExpression: ExpressionSpecification = [
  // Color polygons by the inferred baseline state.
  'match',
  ['get', 'before_state'],
  'vegetation',
  landCoverColors.vegetation,
  'built_up',
  landCoverColors.built_up,
  'water_moisture',
  landCoverColors.water_moisture,
  'bare_sparse',
  landCoverColors.bare_sparse,
  'mixed',
  landCoverColors.mixed,
  landCoverColors.unknown,
]

const afterStateColorExpression: ExpressionSpecification = [
  // Color polygons by the inferred after-change state.
  'match',
  ['get', 'after_state'],
  'vegetation',
  landCoverColors.vegetation,
  'built_up',
  landCoverColors.built_up,
  'water_moisture',
  landCoverColors.water_moisture,
  'bare_sparse',
  landCoverColors.bare_sparse,
  'mixed',
  landCoverColors.mixed,
  landCoverColors.unknown,
]

const magnitudeColorExpression: ExpressionSpecification = [
  // Color by continuous change magnitude when reviewers want intensity rather
  // than class labels.
  'interpolate',
  ['linear'],
  ['coalesce', ['get', 'change_magnitude'], 0],
  0,
  '#f7f7f7',
  0.25,
  '#fdae61',
  0.5,
  '#d7191c',
  1,
  '#67001f',
]

const reliabilityColorExpression: ExpressionSpecification = [
  // Color by publication/review status so reliability summary quantities are
  // visible as map areas, not just sidebar numbers.
  'match',
  ['get', 'reliability'],
  'high',
  reliabilityColors.high,
  'medium',
  reliabilityColors.medium,
  'low',
  reliabilityColors.low,
  reliabilityColors.unknown,
]

const selfSupervisedReviewColorExpression: ExpressionSpecification = [
  // This layer uses outline-style review colors that are separate from the
  // land-cover and reliability palettes. It represents patch diagnostics, not
  // class predictions or measured changed area.
  'match',
  ['get', 'review_priority'],
  'embedding_supported',
  '#0e7490',
  'weak_support_review',
  '#f97316',
  'high_anomaly',
  '#be123c',
  '#64748b',
]

function colorExpressionForMode(
  mode: MapViewProps['vectorColorMode'],
): ExpressionSpecification {
  if (mode === 'reliability') {
    return reliabilityColorExpression
  }
  if (mode === 'baseline') {
    return stateColorExpression
  }
  if (mode === 'after') {
    return afterStateColorExpression
  }
  if (mode === 'magnitude') {
    return magnitudeColorExpression
  }
  return monitoredGroupColorExpression
}

function rasterBounds(layer: RasterLayer): [number, number, number, number] {
  // MapLibre raster tile sources use west/south/east/north bounds, while image
  // fallback sources keep four corner coordinates.
  const lngs = layer.coordinates.map(([lng]) => lng)
  const lats = layer.coordinates.map(([, lat]) => lat)
  return [Math.min(...lngs), Math.min(...lats), Math.max(...lngs), Math.max(...lats)]
}

function removeSatelliteRaster(map: Map) {
  if (map.getLayer('satellite-raster')) {
    map.removeLayer('satellite-raster')
  }
  if (map.getSource('satellite-raster')) {
    map.removeSource('satellite-raster')
  }
}

function addSatelliteRaster(map: Map, rasterLayer: RasterLayer, beforeId?: string) {
  // Preferred path: XYZ tiles in EPSG:3857. This avoids the horizontal drift
  // caused by stretching one large projected PNG over a Web Mercator basemap.
  if (rasterLayer.tile_template) {
    map.addSource('satellite-raster', {
      type: 'raster',
      tiles: [demoPath(rasterLayer.tile_template)],
      tileSize: rasterLayer.tileSize ?? 256,
      minzoom: rasterLayer.minzoom ?? 0,
      maxzoom: rasterLayer.maxzoom ?? 22,
      bounds: rasterBounds(rasterLayer),
    })
  } else {
    map.addSource('satellite-raster', {
      type: 'image',
      url: demoPath(rasterLayer.path),
      coordinates: rasterLayer.coordinates,
    })
  }

  map.addLayer(
    {
      id: 'satellite-raster',
      type: 'raster',
      source: 'satellite-raster',
      paint: {
        'raster-opacity': rasterLayer.opacity,
        'raster-fade-duration': 0,
      },
    },
    beforeId,
  )
}

export function MapView({
  changes,
  selectedLandCover,
  minConfidence,
  activeRasterLayer,
  vectorColorMode,
  selfSupervisedReview,
  showSelfSupervisedReview,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<Map | null>(null)
  const latestChangesRef = useRef(changes)
  const latestRasterRef = useRef(activeRasterLayer)
  const latestColorModeRef = useRef(vectorColorMode)
  const latestSelfSupervisedRef = useRef(selfSupervisedReview)
  const latestShowSelfSupervisedRef = useRef(showSelfSupervisedReview)

  useEffect(() => {
    latestChangesRef.current = changes
  }, [changes])

  useEffect(() => {
    latestRasterRef.current = activeRasterLayer
  }, [activeRasterLayer])

  useEffect(() => {
    latestColorModeRef.current = vectorColorMode
  }, [vectorColorMode])

  useEffect(() => {
    latestSelfSupervisedRef.current = selfSupervisedReview
  }, [selfSupervisedReview])

  useEffect(() => {
    latestShowSelfSupervisedRef.current = showSelfSupervisedReview
  }, [showSelfSupervisedReview])

  useEffect(() => {
    if (!containerRef.current || mapRef.current) {
      return
    }

    // The base map is intentionally simple and open-source; all project value
    // is added through AOI, Sentinel overlays, and change layers.
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: 'OpenStreetMap contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [30.062, -1.944],
      zoom: 11,
    })

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }))
    mapRef.current = map

    map.on('load', async () => {
      let aoi = fallbackAoi

      try {
        aoi = await loadAoi()
      } catch {
        // A visible fallback AOI keeps local UI testing possible even before the
        // backend or static AOI asset is available.
        aoi = fallbackAoi
      }

      map.addSource('aoi', { type: 'geojson', data: aoi })
      map.addLayer({
        id: 'aoi-fill',
        type: 'fill',
        source: 'aoi',
        paint: {
          'fill-color': '#d8ded8',
          'fill-opacity': 0.52,
        },
      })
      map.addLayer({
        id: 'aoi-outline',
        type: 'line',
        source: 'aoi',
        paint: {
          'line-color': '#1f6f4a',
          'line-width': 3,
        },
      })

      const aoiBounds = boundsFromFeatureCollection(aoi)
      if (aoiBounds) {
        scheduleInitialAoiFit(map, aoiBounds)
      }

      const rasterLayer = latestRasterRef.current
      if (rasterLayer) {
        addSatelliteRaster(map, rasterLayer, 'aoi-outline')
      }

      const initialChanges = latestChangesRef.current
      const initialColorExpression = colorExpressionForMode(latestColorModeRef.current)
      const initialReliabilityMode = latestColorModeRef.current === 'reliability'
      const initialSelfSupervised = latestSelfSupervisedRef.current
      const initialShowSelfSupervised = latestShowSelfSupervisedRef.current

      map.addSource('changes', { type: 'geojson', data: initialChanges })
      map.addSource('self-supervised-review', {
        type: 'geojson',
        data: initialSelfSupervised,
      })
      map.addSource('change-points', {
        type: 'geojson',
        data: changePointsFromPolygons(initialChanges),
      })
      map.addLayer({
        id: 'self-supervised-review-fill',
        type: 'fill',
        source: 'self-supervised-review',
        paint: {
          'fill-color': selfSupervisedReviewColorExpression,
          'fill-opacity': initialShowSelfSupervised ? 0.16 : 0,
        },
      })
      map.addLayer({
        id: 'self-supervised-review-outline',
        type: 'line',
        source: 'self-supervised-review',
        paint: {
          'line-color': selfSupervisedReviewColorExpression,
          'line-dasharray': [2, 2],
          'line-opacity': initialShowSelfSupervised ? 0.9 : 0,
          'line-width': [
            'match',
            ['get', 'review_priority'],
            'embedding_supported',
            1.1,
            'weak_support_review',
            1.7,
            'high_anomaly',
            2.2,
            1,
          ],
        },
      })
      map.addLayer({
        id: 'changes-fill',
        type: 'fill',
        source: 'changes',
        paint: {
          'fill-color': initialColorExpression,
          // Confidence controls opacity so lower-certainty changes remain
          // visible but visually less dominant.
          'fill-opacity': initialReliabilityMode ? 0.42 : [
            'interpolate',
            ['linear'],
            ['coalesce', ['get', 'confidence'], 0.75],
            0.75,
            0.58,
            1,
            0.86,
          ],
        },
      })
      map.addLayer({
        id: 'changes-contrast-outline',
        type: 'line',
        source: 'changes',
        paint: {
          // White halo improves readability on top of satellite overlays.
          'line-color': '#ffffff',
          'line-opacity': 0.92,
          'line-width': 4,
        },
      })
      map.addLayer({
        id: 'changes-outline',
        type: 'line',
        source: 'changes',
        paint: {
          'line-color': initialColorExpression,
          'line-opacity': 1,
          'line-width': initialReliabilityMode ? [
            'match',
            ['get', 'reliability'],
            'high',
            2.5,
            'medium',
            3.5,
            'low',
            4.5,
            2,
          ] : [
            'interpolate',
            ['linear'],
            ['coalesce', ['get', 'confidence'], 0.75],
            0.75,
            1.5,
            1,
            3,
          ],
        },
      })
      map.addLayer({
        id: 'change-points',
        type: 'circle',
        source: 'change-points',
        paint: {
          'circle-color': initialColorExpression,
          // Reliability mode is polygon-first so it cannot be confused with
          // land-cover point symbols in the sidebar legend.
          'circle-opacity': initialReliabilityMode ? 0 : 0.92,
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['coalesce', ['get', 'confidence'], 0.75],
            0.75,
            4,
            1,
            8,
          ],
          'circle-stroke-color': '#ffffff',
          'circle-stroke-opacity': 0.95,
          'circle-stroke-width': 1.5,
        },
      })

      map.on('click', 'changes-fill', (event) => {
        const feature = event.features?.[0]
        if (!feature || !event.lngLat) {
          return
        }
        const properties = feature.properties ?? {}
        const { before, after } = transitionParts(
          properties.transition,
          properties.before_state,
          properties.after_state,
        )
        const monitoredSignal = labelForClass(properties.monitored_land_cover)
        const processLabel = labelForChangeType(properties.change_type)
        const reliabilityLabel = labelForReliability(properties.reliability)
        const beforeLabel = labelForClass(before)
        const afterLabel = labelForClass(after)
        const opticalWindow = changeWindowLabel(properties.before_date, properties.after_date)
        const radarWindow = changeWindowLabel(
          properties.radar_before_date,
          properties.radar_after_date,
        )
        const radarSupportRow =
          radarWindow === 'Date window unavailable'
            ? ''
            : `<dt>Radar support</dt><dd>${radarWindow}</dd>`
        // Popup fields mirror the portfolio story: what changed, confidence,
        // when it was observed, magnitude, and area.
        const html = `
          <strong>${processLabel}</strong>
          <span>Observed: ${opticalWindow}</span>
          <span>Before: ${beforeLabel}</span>
          <span>After: ${afterLabel}</span>
          <dl>
            <dt>Monitoring signal</dt><dd>${monitoredSignal}</dd>
            <dt>Process</dt><dd>${processLabel}</dd>
            <dt>Reliability</dt><dd>${reliabilityLabel}</dd>
            <dt>Review note</dt><dd>${String(properties.review_reason ?? 'not assigned').replaceAll('_', ' ')}</dd>
            <dt>Optical dates</dt><dd>${opticalWindow}</dd>
            ${radarSupportRow}
            <dt>Confidence</dt><dd>${Number(properties.confidence ?? 0).toFixed(2)}</dd>
            <dt>Magnitude</dt><dd>${Number(properties.change_magnitude ?? 0).toFixed(2)}</dd>
            <dt>Area</dt><dd>${Math.round(Number(properties.area_m2 ?? 0)).toLocaleString()} m²</dd>
          </dl>
        `
        new maplibregl.Popup({ closeButton: true })
          .setLngLat(event.lngLat)
          .setHTML(html)
          .addTo(map)
      })

      map.on('mouseenter', 'changes-fill', () => {
        map.getCanvas().style.cursor = 'pointer'
      })
      map.on('mouseleave', 'changes-fill', () => {
        map.getCanvas().style.cursor = ''
      })

      map.on('click', 'self-supervised-review-fill', (event) => {
        const feature = event.features?.[0]
        if (!feature || !event.lngLat || !latestShowSelfSupervisedRef.current) {
          return
        }
        const properties = feature.properties ?? {}
        const priority = String(properties.review_priority ?? 'review')
          .replaceAll('_', ' ')
        const interpretation = labelForClass(properties.dominant_interpretation)
        const html = `
          <strong>Self-supervised review patch</strong>
          <span>${priority}</span>
          <dl>
            <dt>Dominant interpretation</dt><dd>${interpretation}</dd>
            <dt>Cluster</dt><dd>${Number(properties.cluster_id ?? 0).toLocaleString()}</dd>
            <dt>Weak-source support</dt><dd>${(Number(properties.weak_source_fraction ?? 0) * 100).toFixed(1)}%</dd>
            <dt>Anomaly score</dt><dd>${Number(properties.anomaly_score ?? 0).toFixed(2)}</dd>
            <dt>Split</dt><dd>${String(properties.split ?? 'unknown')}</dd>
            <dt>Year</dt><dd>${String(properties.year ?? 'unknown')}</dd>
          </dl>
        `
        new maplibregl.Popup({ closeButton: true })
          .setLngLat(event.lngLat)
          .setHTML(html)
          .addTo(map)
      })

      map.on('mouseenter', 'self-supervised-review-fill', () => {
        if (latestShowSelfSupervisedRef.current) {
          map.getCanvas().style.cursor = 'pointer'
        }
      })
      map.on('mouseleave', 'self-supervised-review-fill', () => {
        map.getCanvas().style.cursor = ''
      })
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const source = mapRef.current?.getSource('changes') as GeoJSONSource | undefined
    const pointSource = mapRef.current?.getSource('change-points') as GeoJSONSource | undefined
    // Keep map sources synchronized with sidebar filters without rebuilding the map.
    if (source) {
      source.setData(changes)
    }
    if (pointSource) {
      pointSource.setData(changePointsFromPolygons(changes))
    }
  }, [changes, selectedLandCover, minConfidence])

  useEffect(() => {
    const source = mapRef.current?.getSource('self-supervised-review') as
      | GeoJSONSource
      | undefined
    // Patch footprints are loaded once from the static demo bundle and then
    // toggled visually; source updates keep hot reload and rebuilt exports in sync.
    if (source) {
      source.setData(selfSupervisedReview)
    }
  }, [selfSupervisedReview])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded()) {
      return
    }

    const fillOpacity = showSelfSupervisedReview ? 0.16 : 0
    const lineOpacity = showSelfSupervisedReview ? 0.9 : 0
    if (map.getLayer('self-supervised-review-fill')) {
      map.setPaintProperty('self-supervised-review-fill', 'fill-opacity', fillOpacity)
    }
    if (map.getLayer('self-supervised-review-outline')) {
      map.setPaintProperty('self-supervised-review-outline', 'line-opacity', lineOpacity)
    }
  }, [showSelfSupervisedReview])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded()) {
      return
    }

    const expression = colorExpressionForMode(vectorColorMode)
    const reliabilityMode = vectorColorMode === 'reliability'
    // The same GeoJSON can be restyled by monitored class, before state, after
    // state, magnitude, or reliability without another API request. Reliability
    // uses area fills and outlines only so it is not confused with class symbols.
    if (map.getLayer('changes-fill')) {
      map.setPaintProperty('changes-fill', 'fill-color', expression)
      map.setPaintProperty('changes-fill', 'fill-opacity', reliabilityMode ? 0.42 : [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'confidence'], 0.75],
        0.75,
        0.58,
        1,
        0.86,
      ])
    }
    if (map.getLayer('changes-outline')) {
      map.setPaintProperty('changes-outline', 'line-color', expression)
      map.setPaintProperty('changes-outline', 'line-width', reliabilityMode ? [
        'match',
        ['get', 'reliability'],
        'high',
        2.5,
        'medium',
        3.5,
        'low',
        4.5,
        2,
      ] : [
        'interpolate',
        ['linear'],
        ['coalesce', ['get', 'confidence'], 0.75],
        0.75,
        1.5,
        1,
        3,
      ])
    }
    if (map.getLayer('change-points')) {
      map.setPaintProperty('change-points', 'circle-color', expression)
      map.setPaintProperty('change-points', 'circle-opacity', reliabilityMode ? 0 : 0.92)
    }
  }, [vectorColorMode])

  useEffect(() => {
    const map = mapRef.current
    if (!map?.isStyleLoaded()) {
      return
    }

    if (!activeRasterLayer) {
      // Allow the UI to remove the satellite layer while retaining vector changes.
      removeSatelliteRaster(map)
      return
    }

    removeSatelliteRaster(map)
    addSatelliteRaster(
      map,
      activeRasterLayer,
      map.getLayer('aoi-outline') ? 'aoi-outline' : undefined,
    )
  }, [activeRasterLayer])

  return <div className="map-view" ref={containerRef} />
}
