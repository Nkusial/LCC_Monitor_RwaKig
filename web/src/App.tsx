// Main WebGIS shell: sidebar controls, reliability summary, and MapLibre map.
import './App.css'
import { useEffect, useMemo, useState } from 'react'
import type { FeatureCollection, Geometry } from 'geojson'
import {
  type ApiSummary,
  type Benchmark2026Summary,
  type ChangeSummary,
  type RasterLayer,
  type ReliabilityControlSummary,
  type SelfSupervisedReviewSummary,
  type UnetReliabilitySummary,
  loadChangeGeoJson,
  loadChangeList,
  loadBenchmark2026Summary,
  loadRasterLayerManifest,
  loadReliabilityControlSummary,
  loadSelfSupervisedReviewGeoJson,
  loadSelfSupervisedReviewSummary,
  loadSummary,
  loadUnetReliabilitySummary,
} from './api'
import { MapView } from './map/MapView'
import {
  changeWindowLabel,
  labelForChangeType,
  labelForClass,
  landCoverColors,
  reliabilityLabels,
  rasterLegendForLayer,
  transitionParts,
} from './map/layers'

const emptyChanges: FeatureCollection<Geometry> = {
  type: 'FeatureCollection',
  features: [],
}

const emptyFeatureCollection: FeatureCollection<Geometry> = {
  type: 'FeatureCollection',
  features: [],
}

const searchParams = new URLSearchParams(window.location.search)
// Query parameters are only used for repeatable portfolio screenshots; normal
// users still land on the default Sentinel-2 false-color layer.
const initialLayer = searchParams.get('layer') ?? 's2_false_color_20250220'

function App() {
  const [selectedLandCover, setSelectedLandCover] = useState('all')
  const [minConfidence, setMinConfidence] = useState(0.75)
  const [activeRasterLayerId, setActiveRasterLayerId] = useState(initialLayer)
  const [vectorColorMode, setVectorColorMode] = useState<
    'monitored' | 'baseline' | 'after' | 'magnitude' | 'reliability'
  >('reliability')
  const [changes, setChanges] = useState<ChangeSummary[]>([])
  const [changeFeatures, setChangeFeatures] = useState<FeatureCollection<Geometry>>(emptyChanges)
  const [rasterLayers, setRasterLayers] = useState<RasterLayer[]>([])
  const [unetReliability, setUnetReliability] = useState<UnetReliabilitySummary | null>(null)
  const [benchmark2026, setBenchmark2026] = useState<Benchmark2026Summary | null>(null)
  const [reliabilityControl, setReliabilityControl] =
    useState<ReliabilityControlSummary | null>(null)
  const [selfSupervisedReview, setSelfSupervisedReview] =
    useState<SelfSupervisedReviewSummary | null>(null)
  const [selfSupervisedFeatures, setSelfSupervisedFeatures] =
    useState<FeatureCollection<Geometry>>(emptyFeatureCollection)
  const [showSelfSupervisedReview, setShowSelfSupervisedReview] = useState(false)
  const [summary, setSummary] = useState<ApiSummary>({
    total: 0,
    mean_confidence: null,
    aoi_area_m2: 400_000_000,
    total_area_m2: 0,
    changed_area_m2: 0,
    no_change_area_m2: 400_000_000,
    changed_percent: 0,
    no_change_percent: 100,
    by_monitored_land_cover: {},
    by_monitored_land_cover_area_m2: {},
    by_reliability: {},
    by_reliability_area_m2: {},
  })

  useEffect(() => {
    async function loadInitialData() {
      try {
        // Load core dashboard data first; optional QA reports must never blank
        // the map if one exported JSON file is missing during development.
        const [summaryData, changeData, rasterManifest] = await Promise.all([
          loadSummary(),
          loadChangeList(500),
          loadRasterLayerManifest(),
        ])
        setSummary(summaryData)
        setChanges(changeData)
        setRasterLayers(rasterManifest.layers)
      } catch {
        setChanges([])
      }

      loadUnetReliabilitySummary().then(setUnetReliability).catch(() => undefined)
      loadBenchmark2026Summary().then(setBenchmark2026).catch(() => undefined)
      loadReliabilityControlSummary().then(setReliabilityControl).catch(() => undefined)
      loadSelfSupervisedReviewSummary().then(setSelfSupervisedReview).catch(() => undefined)
      loadSelfSupervisedReviewGeoJson()
        .then(setSelfSupervisedFeatures)
        .catch(() => setSelfSupervisedFeatures(emptyFeatureCollection))
    }

    loadInitialData()
  }, [])

  useEffect(() => {
    async function loadMapChanges() {
      try {
        // Re-query or re-filter change GeoJSON whenever user-facing filters change.
        setChangeFeatures(await loadChangeGeoJson(minConfidence, selectedLandCover))
      } catch {
        setChangeFeatures(emptyChanges)
      }
    }

    loadMapChanges()
  }, [selectedLandCover, minConfidence])

  const filteredCount = changeFeatures.features.length
  const visibleChangedAreaM2 = useMemo(
    () =>
      // Visible area updates with filters, while the full AOI no-change area
      // remains available in the global summary above.
      changeFeatures.features.reduce((total, feature) => {
        const area = Number(feature.properties?.area_m2 ?? 0)
        return Number.isFinite(area) ? total + area : total
      }, 0),
    [changeFeatures],
  )
  const visibleNoChangeAreaM2 = Math.max(summary.aoi_area_m2 - visibleChangedAreaM2, 0)
  const topChanges = useMemo(() => changes.slice(0, 8), [changes])
  const activeRasterLayer = useMemo(
    () => rasterLayers.find((layer) => layer.id === activeRasterLayerId) ?? null,
    [activeRasterLayerId, rasterLayers],
  )
  const activeRasterLegend = useMemo(
    () => rasterLegendForLayer(activeRasterLayer?.id),
    [activeRasterLayer],
  )
  const likelyReliableChangeAreaM2 = useMemo(() => {
    const reliabilityAreas = summary.by_reliability_area_m2 ?? {}
    if (Object.keys(reliabilityAreas).length > 0) {
      return reliabilityAreas.high ?? 0
    }
    return changes.reduce((total, change) => {
      return change.reliability === 'high' ? total + change.area_m2 : total
    }, 0)
  }, [changes, summary.by_reliability_area_m2])
  const needsReviewFraction =
    unetReliability?.summary.mean_candidate_review_fraction_of_valid ??
    reliabilityControl?.current_reliability_snapshot.train_validation_review_fraction ??
    null
  const weakSourceCompatibility =
    unetReliability?.summary.mean_compatible_fraction_in_agreement_zone ??
    reliabilityControl?.current_reliability_snapshot.train_validation_weak_compatibility ??
    null
  const topLikelyChangeType = useMemo(() => {
    if (summary.top_likely_change_type) {
      const [before, after] = summary.top_likely_change_type.split('_to_')
      return before && after ? `${labelForClass(before)} to ${labelForClass(after)}` : summary.top_likely_change_type
    }

    const totals = new Map<string, number>()
    for (const change of changes) {
      const { before, after } = transitionParts(
        change.transition,
        change.before_state,
        change.after_state,
      )
      const key = `${before}_to_${after}`
      totals.set(key, (totals.get(key) ?? 0) + change.area_m2)
    }

    const [transition] = [...totals.entries()].sort((a, b) => b[1] - a[1])[0] ?? []
    if (!transition) {
      return '-'
    }
    const [before, after] = transition.split('_to_')
    return `${labelForClass(before)} to ${labelForClass(after)}`
  }, [changes, summary.top_likely_change_type])
  const selfSupervisedNeedsReviewCount =
    (selfSupervisedReview?.priority_counts.weak_support_review ?? 0) +
    (selfSupervisedReview?.priority_counts.high_anomaly ?? 0)
  const selfSupervisedSupportedCount =
    selfSupervisedReview?.priority_counts.embedding_supported ?? 0

  function formatArea(areaM2: number) {
    // Keep area formatting consistent across metrics, legend, and records.
    return `${(areaM2 / 1_000_000).toLocaleString(undefined, {
      maximumFractionDigits: 1,
    })} km²`
  }

  function reliabilityControlLabel(riskLevel: 'low' | 'moderate' | 'high') {
    // The pipeline uses compact risk levels for sorting and tests. The UI
    // translates them into caution language so users read these as review
    // priorities, not as a failure verdict on the whole prototype.
    const labels = {
      low: 'Lower caution',
      moderate: 'Moderate caution',
      high: 'High caution',
    }
    return labels[riskLevel]
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Monitoring controls">
        <div>
          <p className="eyebrow">Near-real-time monitor</p>
          <h1>Land-cover change</h1>
          <p className="summary">
            Kigali demo AOI with confidence-aware change layers, scene tracking,
            and local-first processing.
          </p>
        </div>

        <section className="control-group" aria-label="Run summary">
          <div className="metric">
            <span>AOI</span>
            <strong>20 x 20 km</strong>
          </div>
          <div className="metric">
            <span>Published</span>
            <strong>{summary.total.toLocaleString()}</strong>
          </div>
          <div className="metric metric-change">
            <span>Changed area</span>
            <strong>{formatArea(summary.changed_area_m2)}</strong>
            <small>{summary.changed_percent.toFixed(1)}% of AOI</small>
          </div>
          <div className="metric">
            <span>Visible</span>
            <strong>{filteredCount.toLocaleString()}</strong>
          </div>
          <div className="metric">
            <span>Visible changed</span>
            <strong>{formatArea(visibleChangedAreaM2)}</strong>
          </div>
          <div className="metric">
            <span>Visible no change</span>
            <strong>{formatArea(visibleNoChangeAreaM2)}</strong>
          </div>
        </section>

        <section className="control-group" aria-label="Legend">
          <h2>Legend</h2>
          <p className="layer-note">
            Satellite overlays are AOI-clipped Web Mercator tiles exported from the
            processed Sentinel GeoTIFF stack.
          </p>
          <div className="legend">
            <span className="no-change">
              <i />
              <b>No change</b>
              <small>{formatArea(summary.no_change_area_m2)}</small>
            </span>
            {Object.entries(landCoverColors)
              .filter(([key]) => key !== 'unknown')
              .map(([key, color]) => (
                <span key={key}>
                  <i style={{ backgroundColor: color }} />
                  <b>{labelForClass(key)}</b>
                  <small>
                    {(summary.by_monitored_land_cover[key] ?? 0).toLocaleString()} ·{' '}
                    {formatArea(summary.by_monitored_land_cover_area_m2[key] ?? 0)}
                  </small>
                </span>
              ))}
          </div>
          <div className="legend reliability-legend" aria-label="Reliability legend">
            {(['high', 'medium', 'low'] as const).map((key) => (
              <span key={key}>
                <i className={`reliability-symbol reliability-symbol-${key}`} />
                <b>{reliabilityLabels[key]}</b>
                <small>{formatArea(summary.by_reliability_area_m2?.[key] ?? 0)}</small>
              </span>
            ))}
          </div>
        </section>

        <section className="control-group" aria-label="Filters">
          <label>
            Satellite-derived layer
            <select
              value={activeRasterLayerId}
              onChange={(event) => setActiveRasterLayerId(event.target.value)}
            >
              {rasterLayers.map((layer) => (
                <option key={layer.id} value={layer.id}>
                  {layer.group}: {layer.title}
                </option>
              ))}
            </select>
          </label>
          {activeRasterLayer ? (
            <p className="layer-note">
              {activeRasterLayer.description}
            </p>
          ) : null}
          {activeRasterLegend ? (
            <div className="raster-legend" aria-label="Selected raster layer legend">
              <strong>{activeRasterLegend.title}</strong>
              {activeRasterLegend.kind === 'gradient' ? (
                <>
                  <span
                    className="raster-gradient"
                    style={{ background: activeRasterLegend.gradient }}
                  />
                  <div className="raster-gradient-labels">
                    <small>{activeRasterLegend.minLabel}</small>
                    <small>{activeRasterLegend.maxLabel}</small>
                  </div>
                </>
              ) : (
                <div className="raster-legend-items">
                  {activeRasterLegend.items.map((item) => (
                    <span key={item.label}>
                      <i style={{ backgroundColor: item.color }} />
                      <small>{item.label}</small>
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : null}
          <fieldset className="segmented-control">
            <legend>Change polygon color</legend>
            <label>
              <input
                type="radio"
                name="vector-color-mode"
                value="reliability"
                checked={vectorColorMode === 'reliability'}
                onChange={() => setVectorColorMode('reliability')}
              />
              Reliability
            </label>
            <label>
              <input
                type="radio"
                name="vector-color-mode"
                value="monitored"
                checked={vectorColorMode === 'monitored'}
                onChange={() => setVectorColorMode('monitored')}
              />
              Monitored
            </label>
            <label>
              <input
                type="radio"
                name="vector-color-mode"
                value="baseline"
                checked={vectorColorMode === 'baseline'}
                onChange={() => setVectorColorMode('baseline')}
              />
              Before
            </label>
            <label>
              <input
                type="radio"
                name="vector-color-mode"
                value="after"
                checked={vectorColorMode === 'after'}
                onChange={() => setVectorColorMode('after')}
              />
              After
            </label>
            <label>
              <input
                type="radio"
                name="vector-color-mode"
                value="magnitude"
                checked={vectorColorMode === 'magnitude'}
                onChange={() => setVectorColorMode('magnitude')}
              />
              Magnitude
            </label>
          </fieldset>
          <label>
            Monitoring signal
            <select
              value={selectedLandCover}
              onChange={(event) => setSelectedLandCover(event.target.value)}
            >
              <option value="all">All monitored groups</option>
              <option value="vegetation">Vegetation</option>
              <option value="built_up">Built-up / impervious</option>
              <option value="water_moisture">Water / wetness signal</option>
              <option value="bare_sparse">Bare or sparse ground</option>
              <option value="mixed">Mixed / uncertain</option>
            </select>
          </label>
          <label>
            Minimum confidence <strong>{minConfidence.toFixed(2)}</strong>
            <input
              type="range"
              min="0.75"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(event) => setMinConfidence(Number(event.target.value))}
            />
          </label>
        </section>

        <section className="control-group reliability-panel" aria-label="Reliability summary">
          <h2>Reliability summary</h2>
          <div className="metric reliability-supported">
            <span>Likely reliable change</span>
            <strong>{formatArea(likelyReliableChangeAreaM2)}</strong>
            <small>High-reliability published change polygons</small>
          </div>
          <div className="metric reliability-review">
            <span>Needs review</span>
            <strong>{needsReviewFraction === null ? '-' : `${(needsReviewFraction * 100).toFixed(1)}%`}</strong>
            <small>Low confidence, high entropy, or weak-source disagreement</small>
          </div>
          <div className="metric reliability-supported">
            <span>Weak-source compatibility</span>
            <strong>{weakSourceCompatibility === null ? '-' : `${(weakSourceCompatibility * 100).toFixed(1)}%`}</strong>
            <small>Agreement with Dynamic World, ESA, OSM, and Sentinel evidence</small>
          </div>
          <div className="metric">
            <span>Mean confidence</span>
            <strong>{summary.mean_confidence ? summary.mean_confidence.toFixed(2) : '-'}</strong>
          </div>
          <div className="metric metric-no-change">
            <span>Stable no-change area</span>
            <strong>{formatArea(summary.no_change_area_m2)}</strong>
          </div>
          <div className="metric metric-stacked">
            <span>Top likely change type</span>
            <strong>{topLikelyChangeType}</strong>
          </div>
          <p className="layer-note">
            {unetReliability?.accuracy_claim ??
              reliabilityControl?.accuracy_claim ??
              'Reliability guidance only; not field-validated accuracy.'}
          </p>
          {unetReliability ? (
            <details className="workflow-panel">
              <summary>Review-zone reasons</summary>
              <div className="review-key">
                <span><i className="low-confidence" />Low confidence</span>
                <span><i className="high-entropy" />High entropy</span>
                <span><i className="weak-disagreement" />Weak-source disagreement</span>
              </div>
              <div className="review-list">
                {unetReliability.top_review_records.slice(0, 3).map((record) => (
                  <span key={record.training_tag}>
                    {record.training_tag.replace('training_', '').replaceAll('_', ' / ')}
                    <b>{(record.candidate_review_fraction_of_valid * 100).toFixed(1)}% review</b>
                    <small>
                      {(record.compatible_fraction_in_agreement_zone * 100).toFixed(1)}% compatible ·{' '}
                      {record.mean_confidence.toFixed(2)} confidence
                    </small>
                  </span>
                ))}
              </div>
            </details>
          ) : null}
          {benchmark2026 ? (
            <details className="workflow-panel">
              <summary>2026 frozen benchmark</summary>
              <p className="layer-note">{benchmark2026.benchmark_caveat}</p>
              <p className="layer-note">No-cheating protocol: {benchmark2026.non_cheating_protocol.allowed}</p>
              <ol>
                {benchmark2026.non_cheating_protocol.forbidden.map((rule) => (
                  <li key={rule}>{rule}</li>
                ))}
              </ol>
            </details>
          ) : null}
          {reliabilityControl ? (
            <details className="workflow-panel">
              <summary>Reliability controls</summary>
              <div className="risk-list">
                {reliabilityControl.risk_drivers.slice(0, 4).map((risk) => (
                  <span key={risk.driver} className={`risk-${risk.risk_level}`}>
                    <b>{risk.driver}</b>
                    <small>{reliabilityControlLabel(risk.risk_level)}</small>
                  </span>
                ))}
              </div>
              <h3>Likely changes without expert samples</h3>
              <ol>
                {reliabilityControl.likely_changes_without_expert_samples.map((item) => (
                  <li key={item.change}>
                    <strong>{item.change}:</strong> {item.expected_direction}
                  </li>
                ))}
              </ol>
            </details>
          ) : null}
          {selfSupervisedReview ? (
            <details className="workflow-panel self-supervised-panel">
              <summary>Self-supervised review</summary>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showSelfSupervisedReview}
                  onChange={(event) => setShowSelfSupervisedReview(event.target.checked)}
                />
                Show patch review overlay
              </label>
              <div className="embedding-metrics">
                <span>
                  Review patches
                  <b>{selfSupervisedReview.feature_count.toLocaleString()}</b>
                </span>
                <span>
                  Embedding-supported
                  <b>{selfSupervisedSupportedCount.toLocaleString()}</b>
                </span>
                <span>
                  Needs review
                  <b>{selfSupervisedNeedsReviewCount.toLocaleString()}</b>
                </span>
                <span>
                  Weak-source compatibility
                  <b>
                    {(selfSupervisedReview.comparison_metrics.weak_source_compatibility * 100).toFixed(1)}%
                  </b>
                </span>
              </div>
              <div className="review-key embedding-review-key">
                <span><i className="embedding-supported" />Embedding-supported patch</span>
                <span><i className="embedding-weak-review" />Needs review: weak support</span>
                <span><i className="embedding-high-anomaly" />High anomaly patch</span>
              </div>
              <p className="layer-note">
                Patch footprints are label-free review evidence, not changed area
                and not field accuracy.
              </p>
            </details>
          ) : null}
        </section>

        <section className="control-group" aria-label="Recent changes">
          <h2>Recent records</h2>
          <div className="change-list">
            {topChanges.map((change) => (
              <article key={change.id} className="change-row">
                <strong>{labelForChangeType(change.change_type)}</strong>
                <span>
                  {(() => {
                    const { before, after } = transitionParts(
                      change.transition,
                      change.before_state,
                      change.after_state,
                    )
                    return `${labelForClass(before)} to ${labelForClass(after)}`
                  })()}
                </span>
                <span>{changeWindowLabel(change.before_date, change.after_date)}</span>
                <small>
                  {labelForClass(change.monitored_land_cover)} ·{' '}
                  {change.confidence.toFixed(2)} confidence ·{' '}
                  {Math.round(change.area_m2).toLocaleString()} m²
                </small>
              </article>
            ))}
          </div>
        </section>
      </aside>

      <section className="map-panel" aria-label="Map">
        <MapView
          changes={changeFeatures}
          selectedLandCover={selectedLandCover}
          minConfidence={minConfidence}
          activeRasterLayer={activeRasterLayer}
          vectorColorMode={vectorColorMode}
          selfSupervisedReview={selfSupervisedFeatures}
          showSelfSupervisedReview={showSelfSupervisedReview}
        />
      </section>
    </main>
  )
}

export default App
