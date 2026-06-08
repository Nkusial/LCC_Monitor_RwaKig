// Shared map styling constants.
import type { FeatureCollection, Polygon } from 'geojson'

export const landCoverColors: Record<string, string> = {
  // These colors are the stable visual contract between change polygons,
  // U-Net class rasters, popup labels, and the sidebar legend.
  vegetation: '#2f855a',
  built_up: '#d9480f',
  water_moisture: '#1c7ed6',
  bare_sparse: '#b7791f',
  mixed: '#7048e8',
  unknown: '#495057',
}

export const landCoverLabels: Record<string, string> = {
  // Labels avoid ambiguous internal names such as "bare_sparse_to_mixed" in
  // the user interface; popups explain before/after states separately.
  vegetation: 'Vegetation',
  built_up: 'Built-up / impervious',
  water_moisture: 'Water / wetness signal',
  bare_sparse: 'Bare or sparse ground',
  mixed: 'Mixed / uncertain',
  unknown: 'Unknown / unclassified',
  managed_vegetation: 'Managed vegetation / cropland',
  natural_vegetation: 'Natural vegetation',
  bare_soil: 'Bare soil',
  water_wetland: 'Water / wetland',
  uncertain_mixed: 'Mixed / uncertain',
}

export const changeTypeLabels: Record<string, string> = {
  vegetation_loss: 'Vegetation loss',
  vegetation_gain: 'Vegetation gain',
  built_up_gain: 'Built-up expansion',
  built_up_loss: 'Built-up reduction',
  moisture_change: 'Moisture-related change',
  radar_change: 'Radar backscatter change',
  mixed_change: 'Mixed or uncertain change',
  spectral_change: 'Spectral change',
}

export const reliabilityColors: Record<string, string> = {
  high: '#005f73',
  medium: '#f59e0b',
  low: '#c51b7d',
  unknown: '#6b7280',
}

export const reliabilityLabels: Record<string, string> = {
  high: 'Likely reliable change',
  medium: 'Needs review',
  low: 'High review priority',
  unknown: 'Reliability not assigned',
}

export type RasterLegendItem = {
  label: string
  color: string
}

export type RasterLegend =
  | {
      kind: 'categorical'
      title: string
      items: RasterLegendItem[]
    }
  | {
      kind: 'gradient'
      title: string
      minLabel: string
      maxLabel: string
      gradient: string
    }

const unetClassLegend: RasterLegend = {
  kind: 'categorical',
  title: 'Predicted land-cover class',
  items: [
    { label: 'Built-up / impervious', color: landCoverColors.built_up },
    { label: 'Vegetation', color: landCoverColors.vegetation },
    { label: 'Bare or sparse ground', color: landCoverColors.bare_sparse },
    { label: 'Water / wetness signal', color: landCoverColors.water_moisture },
    { label: 'Mixed / uncertain', color: landCoverColors.mixed },
  ],
}

const reviewZoneLegend: RasterLegend = {
  kind: 'categorical',
  title: 'Review-zone reason',
  items: [
    { label: 'Low model confidence', color: '#f59e0b' },
    { label: 'High entropy / ambiguity', color: '#dc2626' },
    { label: 'Weak-source disagreement', color: '#7c3aed' },
  ],
}

export function rasterLegendForLayer(layerId: string | null | undefined): RasterLegend | null {
  // Raster layers are generated assets, so the legend is inferred from stable
  // layer IDs rather than duplicated in every manifest record.
  if (!layerId) {
    return null
  }

  if (layerId.includes('review_zones')) {
    return reviewZoneLegend
  }

  if (layerId.includes('dominant_class')) {
    return unetClassLegend
  }

  if (layerId.includes('confidence')) {
    return {
      kind: 'gradient',
      title: 'Confidence',
      minLabel: 'Low',
      maxLabel: 'High',
      gradient: 'linear-gradient(90deg, #f7f7f7, #fee090, #fdae61, #a50026)',
    }
  }

  if (layerId.includes('entropy')) {
    return {
      kind: 'gradient',
      title: 'Uncertainty / entropy',
      minLabel: 'More certain',
      maxLabel: 'More uncertain',
      gradient: 'linear-gradient(90deg, #1a9641, #ffffbf, #fdae61, #d7191c)',
    }
  }

  if (layerId.includes('ndvi')) {
    return {
      kind: 'gradient',
      title: 'NDVI vegetation signal',
      minLabel: 'Bare / low vegetation',
      maxLabel: 'Dense vegetation',
      gradient: 'linear-gradient(90deg, #744b2a, #eee28f, #238443, #00441b)',
    }
  }

  if (layerId.includes('ndwi')) {
    return {
      kind: 'gradient',
      title: 'NDWI moisture signal',
      minLabel: 'Dry / low moisture',
      maxLabel: 'Wet / water signal',
      gradient: 'linear-gradient(90deg, #795548, #e8e2ce, #5dade2, #154360)',
    }
  }

  if (layerId.includes('ndbi')) {
    return {
      kind: 'gradient',
      title: 'NDBI built-up signal',
      minLabel: 'Vegetated / low built-up',
      maxLabel: 'Built-up / impervious',
      gradient: 'linear-gradient(90deg, #238443, #f4f1de, #d6604d, #7b3240)',
    }
  }

  if (layerId.includes('delta')) {
    return {
      kind: 'gradient',
      title: 'Change magnitude',
      minLabel: 'Decrease',
      maxLabel: 'Increase',
      gradient: 'linear-gradient(90deg, #762a83, #f7f7f7, #1b7837)',
    }
  }

  if (layerId.includes('s1') || layerId.includes('ratio')) {
    return {
      kind: 'gradient',
      title: 'Radar backscatter / structure',
      minLabel: 'Lower signal',
      maxLabel: 'Higher signal',
      gradient: 'linear-gradient(90deg, #263238, #546e7a, #b0bec5, #ffffff)',
    }
  }

  if (layerId.includes('clusters')) {
    return {
      kind: 'categorical',
      title: 'Unsupervised cluster IDs',
      items: [
        { label: 'Cluster 1', color: '#238443' },
        { label: 'Cluster 2', color: '#fdae61' },
        { label: 'Cluster 3', color: '#d8b365' },
        { label: 'Cluster 4', color: '#80cdc1' },
        { label: 'Cluster 5+', color: '#5e4fa2' },
      ],
    }
  }

  if (layerId.includes('false_color')) {
    return {
      kind: 'gradient',
      title: 'False-color composite',
      minLabel: 'Darker surface',
      maxLabel: 'Brighter vegetation/surface',
      gradient: 'linear-gradient(90deg, #1f2933, #6b705c, #b7b7a4, #f8f9fa)',
    }
  }

  return null
}

export function labelForClass(value: unknown) {
  const key = typeof value === 'string' ? value : 'unknown'
  if (!key) {
    return landCoverLabels.unknown
  }
  return landCoverLabels[key] ?? key.replaceAll('_', ' ')
}

export function labelForChangeType(value: unknown) {
  const key = typeof value === 'string' ? value : 'mixed_change'
  if (!key) {
    return changeTypeLabels.mixed_change
  }
  return changeTypeLabels[key] ?? key.replaceAll('_', ' ')
}

export function labelForReliability(value: unknown) {
  const key = typeof value === 'string' ? value : 'unknown'
  return reliabilityLabels[key] ?? reliabilityLabels.unknown
}

export function dateLabel(value: unknown) {
  if (typeof value !== 'string' || value.length === 0) {
    return 'unknown date'
  }
  return value
}

export function changeWindowLabel(beforeDate: unknown, afterDate: unknown) {
  const before = dateLabel(beforeDate)
  const after = dateLabel(afterDate)
  if (before === 'unknown date' && after === 'unknown date') {
    return 'Date window unavailable'
  }
  return `${before} to ${after}`
}

export function transitionParts(
  transition: unknown,
  beforeState: unknown,
  afterState: unknown,
) {
  if (
    typeof beforeState === 'string' &&
    beforeState.length > 0 &&
    typeof afterState === 'string' &&
    afterState.length > 0
  ) {
    return { before: beforeState, after: afterState }
  }

  if (typeof transition === 'string' && transition.includes('_to_')) {
    const [before, after] = transition.split('_to_')
    return { before, after }
  }

  return { before: 'unknown', after: 'unknown' }
}

export const fallbackAoi: FeatureCollection<Polygon> = {
  // Used only if the API/static AOI file cannot be loaded. It still mirrors the
  // master-grid footprint so emergency rendering does not reintroduce AOI drift.
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        name: 'Kigali 20km demo AOI',
      },
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [29.971674588098654, -1.8536766484566063],
            [30.152387520306693, -1.8536766484566063],
            [30.152387520306693, -2.0344188710956463],
            [29.971674588098654, -2.0344188710956463],
            [29.971674588098654, -1.8536766484566063],
          ],
        ],
      },
    },
  ],
}
