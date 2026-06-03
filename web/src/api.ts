// API and static-demo data access for the WebGIS.
//
// During local development the app talks to FastAPI/PostGIS. On GitHub Pages,
// where no backend is running, the same UI falls back to exported demo JSON and
// Web Mercator raster tiles under `public/demo`.
import type { FeatureCollection, Geometry, Polygon } from 'geojson'

export type ChangeSummary = {
  id: string
  change_type: string
  confidence: number
  area_m2: number
  monitored_land_cover?: string | null
  before_state?: string | null
  after_state?: string | null
  transition?: string | null
  before_date?: string | null
  after_date?: string | null
  radar_before_date?: string | null
  radar_after_date?: string | null
  reliability?: string | null
}

export type ApiSummary = {
  total: number
  mean_confidence: number | null
  aoi_area_m2: number
  total_area_m2: number
  changed_area_m2: number
  no_change_area_m2: number
  changed_percent: number
  no_change_percent: number
  by_monitored_land_cover: Record<string, number>
  by_monitored_land_cover_area_m2: Record<string, number>
  by_reliability: Record<string, number>
  by_reliability_area_m2: Record<string, number>
  top_likely_change_type?: string | null
  top_likely_change_area_m2?: number
}

export type RasterLayer = {
  id: string
  title: string
  group: string
  kind: 'raster_overlay'
  path: string
  tile_template?: string
  minzoom?: number
  maxzoom?: number
  tileSize?: number
  coordinates: [[number, number], [number, number], [number, number], [number, number]]
  opacity: number
  description: string
  source: string
  satelliteDerived: boolean
}

export type RasterLayerManifest = {
  generated_from: string
  grid_alignment?: {
    status: string
    reference: string
    checks: string[]
    display_aoi: string
  }
  display_alignment?: {
    status: string
    display_crs: string
    analysis_crs: string
    tile_scheme?: string
    tile_size?: number
    minzoom?: number
    maxzoom?: number
    reason: string
  }
  note: string
  layers: RasterLayer[]
}

export type GeospatialAlignmentSummary = {
  phase: string
  status: 'passed' | 'failed'
  master_grid: {
    path: string
    crs: string
    width: number
    height: number
    bounds: number[]
    transform: number[]
    tolerance: number
  }
  checked_raster_count: number
  checked_vector_count: number
  raster_mismatch_count: number
  vector_mismatch_count: number
  raster_family_counts: Record<string, number>
  skipped_raw_source_rasters: string[]
  interpretation: string
}

export type CoregistrationQaSummary = {
  phase: string
  status: 'passed' | 'warning' | 'failed'
  method: string
  reference: string
  master_grid: {
    path: string
    crs: string
    resolution_m: number
  }
  checked_target_count: number
  grid_failure_count: number
  high_shift_warning_count: number
  max_diagnostic_shift_m: number
  records: Array<{
    family: string
    label: string
    path: string
    grid_aligned: boolean
    diagnostic_shift_pixels: {
      dx: number
      dy: number
      max_abs: number
    }
    diagnostic_shift_m: number
    gradient_correlation: number | null
    interpretation: string
  }>
  limitations: string[]
  recommended_production_method: string[]
}

export type ValidationSummary = {
  phase: string
  validation_type: string
  ground_truth_available: boolean
  accuracy_claim: string
  score_label?: string
  score_scope_note?: string
  source_report?: string
  silhouette_score: number | null
  best_experiment: {
    monitoring_tag: string
    feature_set: string
    features: string[]
    reduction: string
    mask_mode: string
    cluster_count: number
    valid_pixel_count: number
    sample_size: number
    evaluation_size: number
    silhouette_score: number
  }
  cluster_profiles: Array<{
    cluster_id: number
    area_km2: number
    inferred_land_cover: string
  }>
  monitored_group_cluster_alignment: {
    status: string
    by_monitored_land_cover: Record<
      string,
      {
        sampled_feature_count: number
        dominant_cluster: number
        dominant_cluster_share: number
      }
    >
  }
  reference_label_workflow: {
    status: string
    planned_steps: string[]
  }
}

export type UnetReliabilitySummary = {
  phase: string
  purpose: string
  accuracy_claim: string
  weak_sources_used: string[]
  thresholds: {
    high_entropy: number
    low_confidence: number
    minimum_agreement_sources: number
  }
  summary: {
    record_count: number
    mean_compatible_fraction_in_agreement_zone: number
    mean_candidate_review_fraction_of_valid: number
    highest_review_priority: string
  }
  limitations: string[]
  top_review_records: Array<{
    training_tag: string
    year: number
    split: string
    compatible_fraction_in_agreement_zone: number
    candidate_review_fraction_of_valid: number
    mean_confidence: number
    mean_entropy: number
    coverage_fraction: number
  }>
}

export type Benchmark2026Summary = {
  phase: string
  title: string
  status: string
  accuracy_claim: string
  benchmark_caveat: string
  weak_sources_used: string[]
  thresholds: {
    high_entropy: number
    low_confidence: number
    minimum_agreement_sources: number
  }
  summary: {
    record_count: number
    mean_compatible_fraction_in_agreement_zone: number
    mean_candidate_review_fraction_of_valid: number
    highest_review_priority: string
  }
  record: {
    training_tag: string
    year: number
    split: string
    weak_agreement_zone_fraction_of_valid: number
    compatible_fraction_in_agreement_zone: number
    candidate_review_fraction_of_valid: number
    mean_confidence: number
    mean_entropy: number
    coverage_fraction: number
  }
  limitations: string[]
  non_cheating_protocol: {
    allowed: string
    forbidden: string[]
  }
}

export type ReliabilityControlSummary = {
  phase: string
  purpose: string
  accuracy_claim: string
  no_cheating_boundary: {
    frozen_test_set: string
    allowed: string[]
    forbidden: string[]
  }
  current_reliability_snapshot: {
    train_validation_mean_confidence: number
    train_validation_mean_entropy: number
    train_validation_review_fraction: number
    train_validation_weak_compatibility: number
    frozen_2026_mean_confidence: number
    frozen_2026_mean_entropy: number
    frozen_2026_review_fraction: number
    frozen_2026_weak_compatibility: number
  }
  driver_metrics: Record<string, number>
  risk_drivers: Array<{
    driver: string
    risk_level: 'low' | 'moderate' | 'high'
    evidence: string
    safe_action: string
  }>
  likely_changes_without_expert_samples: Array<{
    change: string
    expected_direction: string
    why: string
  }>
  recommended_next_controls: string[]
}

export type SelfSupervisedReviewSummary = {
  phase: string
  purpose: string
  accuracy_claim: string
  source_report: string
  feature_count: number
  priority_counts: Record<string, number>
  priority_area_m2: Record<string, number>
  legend: Record<string, string>
  comparison_metrics: {
    silhouette_score: number
    weak_source_compatibility: number
    review_burden_fraction: number
    high_anomaly_fraction: number
    mean_cluster_ambiguity: number
    coherent_cluster_fraction: number
  }
  no_cheating_protocol: {
    excluded_years_from_fit: number[]
    fit_inputs: string
    weak_labels_role: string
    frozen_test_boundary: string
  }
}

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined
// Use the exported demo bundle by default so the WebGIS is reproducible even
// when a local FastAPI process is stopped, empty, or pointed at a stale DB.
// Set VITE_API_BASE_URL=http://127.0.0.1:8000 when intentionally testing the API.
export const apiBaseUrl = configuredApiBaseUrl ?? ''

export function demoPath(path: string) {
  // BASE_URL keeps assets working both at localhost and under the GitHub Pages
  // repository subpath.
  return `${import.meta.env.BASE_URL}demo/${path}`.replace(/\/{2,}/g, '/')
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`)
  }
  return response.json() as Promise<T>
}

async function fetchApiJson<T>(path: string): Promise<T> {
  if (!apiBaseUrl) {
    throw new Error('No API base URL configured')
  }
  return fetchJson<T>(`${apiBaseUrl}${path}`)
}

function featureToChangeSummary(feature: GeoJSON.Feature<Geometry>): ChangeSummary {
  // Static GeoJSON stores the same attributes as PostGIS, but the sidebar wants
  // a compact list shape rather than full geometries.
  const properties = feature.properties ?? {}
  return {
    id: String(properties.id ?? feature.id ?? crypto.randomUUID()),
    change_type: String(properties.change_type ?? 'change'),
    confidence: Number(properties.confidence ?? 0),
    area_m2: Number(properties.area_m2 ?? 0),
    monitored_land_cover: String(properties.monitored_land_cover ?? 'unknown'),
    before_state: String(properties.before_state ?? ''),
    after_state: String(properties.after_state ?? ''),
    transition: String(properties.transition ?? ''),
    before_date: String(properties.before_date ?? ''),
    after_date: String(properties.after_date ?? ''),
    radar_before_date: String(properties.radar_before_date ?? ''),
    radar_after_date: String(properties.radar_after_date ?? ''),
    reliability: String(properties.reliability ?? ''),
  }
}

function filterChanges(
  changes: FeatureCollection<Geometry>,
  minConfidence: number,
  monitoredLandCover: string,
) {
  // Mirror the backend filters so the hosted static demo behaves like the API.
  return {
    ...changes,
    features: changes.features.filter((feature) => {
      const properties = feature.properties ?? {}
      const confidence = Number(properties.confidence ?? 0)
      const group = String(properties.monitored_land_cover ?? 'unknown')
      return (
        confidence >= minConfidence &&
        (monitoredLandCover === 'all' || group === monitoredLandCover)
      )
    }),
  }
}

export async function loadSummary(): Promise<ApiSummary> {
  try {
    return await fetchApiJson<ApiSummary>('/changes/summary')
  } catch {
    // Static fallback keeps the hosted portfolio app interactive without a server.
    return fetchJson<ApiSummary>(demoPath('summary.json'))
  }
}

export async function loadAoi(): Promise<FeatureCollection<Polygon>> {
  try {
    return await fetchApiJson<FeatureCollection<Polygon>>('/aoi')
  } catch {
    return fetchJson<FeatureCollection<Polygon>>(demoPath('aoi.geojson'))
  }
}

export async function loadChangeList(limit = 500): Promise<ChangeSummary[]> {
  try {
    return await fetchApiJson<ChangeSummary[]>(`/changes?limit=${limit}`)
  } catch {
    const demoChanges = await fetchJson<FeatureCollection<Geometry>>(
      demoPath('changes.geojson'),
    )
    return demoChanges.features.slice(0, limit).map(featureToChangeSummary)
  }
}

export async function loadChangeGeoJson(
  minConfidence: number,
  monitoredLandCover: string,
): Promise<FeatureCollection<Geometry>> {
  const params = new URLSearchParams({
    limit: '1500',
    min_confidence: minConfidence.toString(),
    monitored_land_cover: monitoredLandCover,
  })

  try {
    return await fetchApiJson<FeatureCollection<Geometry>>(
      `/changes/geojson?${params.toString()}`,
    )
  } catch {
    const demoChanges = await fetchJson<FeatureCollection<Geometry>>(
      demoPath('changes.geojson'),
    )
    return filterChanges(demoChanges, minConfidence, monitoredLandCover)
  }
}

export async function loadRasterLayerManifest(): Promise<RasterLayerManifest> {
  return fetchJson<RasterLayerManifest>(demoPath('raster_layers.json'))
}

export async function loadGeospatialAlignmentSummary(): Promise<GeospatialAlignmentSummary> {
  return fetchJson<GeospatialAlignmentSummary>(demoPath('geospatial_alignment_report.json'))
}

export async function loadCoregistrationQaSummary(): Promise<CoregistrationQaSummary> {
  return fetchJson<CoregistrationQaSummary>(demoPath('coregistration_qa_report.json'))
}

export async function loadValidationSummary(): Promise<ValidationSummary> {
  return fetchJson<ValidationSummary>(demoPath('validation_summary.json'))
}

export async function loadUnetReliabilitySummary(): Promise<UnetReliabilitySummary> {
  return fetchJson<UnetReliabilitySummary>(demoPath('unet_reliability_summary.json'))
}

export async function loadBenchmark2026Summary(): Promise<Benchmark2026Summary> {
  return fetchJson<Benchmark2026Summary>(demoPath('benchmark_2026_summary.json'))
}

export async function loadReliabilityControlSummary(): Promise<ReliabilityControlSummary> {
  return fetchJson<ReliabilityControlSummary>(
    demoPath('reliability_uncertainty_control_summary.json'),
  )
}

export async function loadSelfSupervisedReviewSummary(): Promise<SelfSupervisedReviewSummary> {
  return fetchJson<SelfSupervisedReviewSummary>(
    demoPath('self_supervised_review_layer_summary.json'),
  )
}

export async function loadSelfSupervisedReviewGeoJson(): Promise<FeatureCollection<Geometry>> {
  return fetchJson<FeatureCollection<Geometry>>(
    demoPath('self_supervised_review_patches.geojson'),
  )
}
