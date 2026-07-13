#!/usr/bin/env python3
"""Generate a dependency-free static HTML viewer for quality harness outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

SCRYFALL_BULK_API = "https://api.scryfall.com/bulk-data/default-cards"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def scryfall_cache_dir() -> Path:
  root = os.environ.get("COLLECTORVISION_CACHE")
  if root:
    return Path(root) / "e2e-youtube-quality" / "scryfall"
  return Path.home() / ".cache" / "collectorvision" / "e2e-youtube-quality" / "scryfall"


def download_scryfall_bulk(cache_dir: Path) -> Path:
  cache_dir.mkdir(parents=True, exist_ok=True)
  bulk_path = cache_dir / "default_cards.json"
  if bulk_path.exists() and bulk_path.stat().st_size > 0:
    return bulk_path

  req = urllib.request.Request(
    SCRYFALL_BULK_API,
    headers={
      "Accept": "application/json",
      "User-Agent": "CollectorVision/0.1 (contact: local e2e quality viewer)",
    },
  )
  with urllib.request.urlopen(req, timeout=30) as response:
    bulk_meta = json.loads(response.read().decode("utf-8"))

  download_uri = bulk_meta["download_uri"]
  tmp_path = bulk_path.with_suffix(".json.tmp")
  print(f"Downloading Scryfall bulk data -> {bulk_path}")
  urllib.request.urlretrieve(download_uri, tmp_path)
  tmp_path.replace(bulk_path)
  (cache_dir / "default_cards_meta.json").write_text(json.dumps(bulk_meta, indent=2), encoding="utf-8")
  return bulk_path


def collect_card_ids(records_by_run: dict[str, list[dict[str, Any]]], seen_by_run: dict[str, list[dict[str, Any]]]) -> set[str]:
  ids: set[str] = set()
  for rows in seen_by_run.values():
    for row in rows:
      ids.update(value for value in (row.get("card_id"), row.get("oracle_id")) if value)
  for records in records_by_run.values():
    for record in records:
      ids.update(value for value in (record.get("best_card_id"),) if value)
      for hit in record.get("top_k") or []:
        ids.update(value for value in (hit.get("card_id"), hit.get("oracle_id")) if value)
  return ids


def load_scryfall_names(card_ids: set[str], enabled: bool) -> dict[str, str]:
  if not enabled or not card_ids:
    return {}
  try:
    bulk_path = download_scryfall_bulk(scryfall_cache_dir())
    cards = json.loads(bulk_path.read_text(encoding="utf-8"))
  except Exception as exc:
    print(f"Warning: could not load Scryfall bulk names: {exc}")
    return {}

  names: dict[str, str] = {}
  remaining = set(card_ids)
  for card in cards:
    card_id = card.get("id")
    oracle_id = card.get("oracle_id")
    name = card.get("name")
    if not name:
      continue
    if card_id in remaining:
      set_code = str(card.get("set") or "").upper()
      collector_number = card.get("collector_number") or ""
      if set_code and collector_number:
        suffix = f" [{set_code}] #{collector_number}"
      elif set_code:
        suffix = f" [{set_code}]"
      elif collector_number:
        suffix = f" #{collector_number}"
      else:
        suffix = ""
      names[card_id] = name + suffix
      remaining.discard(card_id)
    if oracle_id in remaining:
      names[oracle_id] = name
      remaining.discard(oracle_id)
    if not remaining:
      break
  return names


def write_viewer(run_dir: Path, output_path: Path | None = None, card_names: bool = True) -> Path:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = list(manifest.get("runs", {}).keys())
    if not labels:
        raise SystemExit(f"No runs found in manifest: {manifest_path}")

    records_by_run = {label: load_jsonl(run_dir / "runs" / label / "frames.jsonl") for label in labels}
    seen_by_run = {label: load_csv(run_dir / "runs" / label / "seen_cards.csv") for label in labels}
    names_by_id = load_scryfall_names(collect_card_ids(records_by_run, seen_by_run), card_names)
    frame_count = max((len(records) for records in records_by_run.values()), default=0)
    payload = {
        "manifest": manifest,
        "labels": labels,
        "recordsByRun": records_by_run,
        "seenByRun": seen_by_run,
        "namesById": names_by_id,
        "frameCount": frame_count,
    }
    output_path = output_path or run_dir / "viewer.html"
    output_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {output_path}")
    return output_path


def render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload).replace("</", "<\\/")
    title = f"CollectorVision Quality Viewer - {payload['manifest'].get('video_id', '')}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f7f5ef;
      --ink: #1f2528;
      --muted: #667074;
      --line: #d8d1c3;
      --panel: #fffdf8;
      --accent: #0f766e;
      --warn: #b42318;
      --soft-accent: #dff3ee;
      --soft-warn: #fae3df;
      --soft-neutral: #ece7dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ position: sticky; top: 0; z-index: 2; background: rgba(247, 245, 239, 0.96); border-bottom: 1px solid var(--line); padding: 14px 18px; }}
    h1 {{ margin: 0 0 10px; font-size: 20px; font-weight: 700; }}
    .controls {{ display: grid; grid-template-columns: minmax(180px, 1fr) auto auto; gap: 12px; align-items: center; }}
    .nav {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }}
    .threshold {{ display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 13px; }}
    input[type="number"] {{ width: 82px; padding: 5px 7px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--ink); }}
    input[type="range"] {{ width: 100%; }}
    button {{ border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--ink); padding: 6px 10px; cursor: pointer; }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    button:disabled {{ cursor: not-allowed; color: var(--muted); opacity: 0.5; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px; }}
    .runs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; align-items: start; }}
    .metric-charts {{ margin-top: 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric-charts h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; margin-top: 12px; }}
    .chart-panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fffaf1; }}
    .chart-panel h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .metric-svg {{ width: 100%; height: 170px; display: block; background: #fffdf8; border: 1px solid var(--line); border-radius: 6px; }}
    .chart-legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }}
    .legend-swatch {{ display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 4px; }}
    .card-summary {{ margin-top: 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .card-summary h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .frame-comparison {{ margin-top: 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .frame-comparison h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .comparison-controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 10px 0; }}
    .comparison-controls button.active {{ border-color: var(--accent); background: var(--soft-accent); color: var(--accent); font-weight: 700; }}
    .comparison-table-wrap {{ max-height: 620px; overflow: auto; margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; }}
    .comparison-table {{ margin: 0; }}
    .comparison-table th {{ position: sticky; top: 0; background: var(--panel); z-index: 1; }}
    .comparison-table tr.agree {{ background: var(--soft-accent); }}
    .comparison-table tr.disagree {{ background: var(--soft-warn); }}
    .comparison-table tr.condensed {{ background: var(--soft-neutral); color: var(--muted); }}
    .comparison-card {{ min-width: 260px; }}
    .summary-tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
    .summary-tabs button.active {{ border-color: var(--accent); background: var(--soft-accent); color: var(--accent); font-weight: 700; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-top: 12px; }}
    .summary-panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fffaf1; }}
    .summary-panel h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .card-list {{ max-height: 260px; overflow: auto; display: flex; flex-direction: column; gap: 4px; }}
    .card-pill {{ display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; padding: 5px 7px; border-radius: 6px; border: 1px solid var(--line); background: var(--soft-neutral); font-size: 12px; }}
    .card-pill.shared {{ background: var(--soft-accent); border-color: #9ed4c7; }}
    .card-pill.only {{ background: var(--soft-warn); border-color: #efb1a8; }}
    .tag {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; color: var(--muted); }}
    .summary-table-wrap {{ max-height: 520px; overflow: auto; margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; }}
    .summary-table {{ margin: 0; }}
    .summary-table th {{ position: sticky; top: 0; background: var(--panel); z-index: 1; }}
    tr.shared {{ background: var(--soft-accent); }}
    tr.mismatch {{ background: var(--soft-warn); }}
    .run {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .run h2 {{ margin: 0 0 8px; font-size: 16px; }}
    .status {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; color: var(--muted); font-size: 13px; }}
    .best {{ color: var(--accent); font-weight: 700; }}
    .card-title {{ display: block; color: var(--ink); font-size: 13px; font-weight: 600; margin-top: 3px; }}
    .disagree .best {{ color: var(--warn); }}
    .thresholded {{ color: var(--warn); }}
    img {{ max-width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid var(--line); background: #eee; }}
    .media {{ display: grid; grid-template-columns: 1fr minmax(120px, 180px); gap: 10px; align-items: start; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ text-align: left; padding: 6px 4px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    a {{ color: var(--accent); }}
    @media (max-width: 760px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .media {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <div class="controls">
      <input id="frame" type="range" min="1" max="1" value="1">
      <strong id="frameLabel">Frame 1</strong>
      <span class="meta" id="timeLabel"></span>
    </div>
    <div class="nav">
      <label class="threshold">sharpness threshold <input id="sharpnessThreshold" type="number" min="0" max="1" step="0.001" value="0.02"></label>
      <label class="threshold">corner quality threshold <input id="cornerQualityThreshold" type="number" min="0" max="1" step="0.01" value="0"></label>
      <label class="threshold">match score threshold <input id="scoreThreshold" type="number" min="-1" max="1" step="0.01" value="0"></label>
      <button id="prevDisagreement" type="button">Prev disagreement</button>
      <button id="nextDisagreement" type="button">Next disagreement</button>
      <button id="strongestAgreement" type="button">Strongest agreement</button>
      <button id="strongestDisagreement" type="button">Strongest disagreement</button>
      <span class="meta" id="agreementStats"></span>
    </div>
    <div class="meta" id="summary"></div>
  </header>
  <main>
    <div class="runs" id="runs"></div>
    <section class="metric-charts" id="metricCharts"></section>
    <section class="frame-comparison" id="frameComparison"></section>
    <section class="card-summary" id="cardSummary"></section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const frame = document.getElementById('frame');
    const frameLabel = document.getElementById('frameLabel');
    const timeLabel = document.getElementById('timeLabel');
    const thresholdInput = document.getElementById('sharpnessThreshold');
    const cornerQualityThresholdInput = document.getElementById('cornerQualityThreshold');
    const scoreThresholdInput = document.getElementById('scoreThreshold');
    const agreementStats = document.getElementById('agreementStats');
    const prevDisagreement = document.getElementById('prevDisagreement');
    const nextDisagreement = document.getElementById('nextDisagreement');
    const strongestAgreement = document.getElementById('strongestAgreement');
    const strongestDisagreement = document.getElementById('strongestDisagreement');
    const summary = document.getElementById('summary');
    const runs = document.getElementById('runs');
    const metricCharts = document.getElementById('metricCharts');
    const frameComparison = document.getElementById('frameComparison');
    const cardSummary = document.getElementById('cardSummary');
    frame.max = Math.max(1, data.frameCount);
    const configuredMinScores = Object.values(data.manifest.runs || {{}}).map(run => Number(run.min_score)).filter(Number.isFinite);
    const configuredMinCornerQuality = Object.values(data.manifest.runs || {{}}).map(run => Number(run.min_corner_quality)).filter(Number.isFinite);
    scoreThresholdInput.value = configuredMinScores.length ? String(Math.max(...configuredMinScores)) : '0';
    cornerQualityThresholdInput.value = configuredMinCornerQuality.length ? String(Math.max(...configuredMinCornerQuality)) : '0';
    summary.textContent = `${{data.labels.length}} runs, ${{data.frameCount}} frames, fps=${{data.manifest.fps}}`;
    let indexCache = {{ disagreements: [], strongestAgreement: null, strongestDisagreement: null }};
    let summaryMode = 'oracle';
    let comparisonMode = 'oracle';
    let condenseAgreements = true;

    function link(path, text) {{
      return path ? `<a href="${{path}}">${{text}}</a>` : '';
    }}
    function fmt(value, digits = 3) {{
      return value === null || value === undefined ? '' : Number(value).toFixed(digits);
    }}
    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function toNumber(value) {{
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }}
    function cardKey(row, mode = summaryMode) {{
      if (mode === 'oracle') return row?.oracle_id || row?.card_id || '';
      return row?.card_id || '';
    }}
    function groupLabel(mode = summaryMode) {{
      return mode === 'oracle' ? 'oracle / secondary ID' : 'exact card ID';
    }}
    function cardDisplay(id) {{
      if (!id) return '';
      return data.namesById?.[id] || id;
    }}
    function cardTitle(id) {{
      const value = cardDisplay(id);
      return value ? `<span class="card-title">${{escapeHtml(value)}}</span>` : '';
    }}
    function recordOracleId(record) {{
      return record?.top_k?.[0]?.oracle_id || '';
    }}
    function compareKeyForRecord(record, effective, mode) {{
      if (!effective.present) return 'NO_MATCH';
      if (mode === 'oracle') return recordOracleId(record) || effective.id || 'NO_MATCH';
      return effective.id || 'NO_MATCH';
    }}
    function rowTime(row) {{
      return toNumber(row?.first_seen_timestamp_sec) ?? Number.POSITIVE_INFINITY;
    }}
    function rowFrame(row) {{
      return toNumber(row?.first_seen_frame) ?? Number.POSITIVE_INFINITY;
    }}
    function jumpToFrameNumber(frameNumber) {{
      const parsed = Number(frameNumber);
      if (!Number.isFinite(parsed)) return;
      jumpTo(Math.max(0, Math.min(data.frameCount - 1, parsed - 1)));
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}
    function sharpnessThreshold() {{
      const value = Number(thresholdInput.value);
      return Number.isFinite(value) ? value : 0.02;
    }}
    function scoreThreshold() {{
      const value = Number(scoreThresholdInput.value);
      return Number.isFinite(value) ? value : 0;
    }}
    function cornerQualityThreshold() {{
      const value = Number(cornerQualityThresholdInput.value);
      return Number.isFinite(value) ? value : 0;
    }}
    function cornerQuality(record) {{
      const value = record?.corner_quality?.score;
      return value === null || value === undefined ? null : Number(value);
    }}
    function metricLine(label, value, threshold, digits = 3) {{
      const display = value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : fmt(value, digits);
      return `${{label}} ${{display}} / ${{fmt(threshold, digits)}}`;
    }}
    function thresholdMisses(record, effective) {{
      if (!record) return ['missing record'];
      const misses = [];
      const rawPresent = Boolean(record?.detector_card_present ?? record?.card_present);
      const sharpness = record?.sharpness === null || record?.sharpness === undefined ? null : Number(record.sharpness);
      const quality = cornerQuality(record);
      const score = record?.best_score === null || record?.best_score === undefined ? null : Number(record.best_score);
      if (!rawPresent) misses.push('no detector-positive corners');
      if (sharpness !== null && sharpness < sharpnessThreshold()) misses.push(metricLine('sharpness', sharpness, sharpnessThreshold(), 4));
      if (quality !== null && quality < cornerQualityThreshold()) misses.push(metricLine('quality', quality, cornerQualityThreshold(), 2));
      if (score === null || score < scoreThreshold()) misses.push(metricLine('score', score, scoreThreshold(), 3));
      if (rawPresent && !record.best_card_id) misses.push('no top match');
      if (!misses.length && effective?.thresholded) misses.push('filtered by current thresholds');
      return misses;
    }}
    function missText(record, effective) {{
      return thresholdMisses(record, effective).join('; ');
    }}
    function effectiveRecord(record, threshold, minScore, minCornerQuality) {{
      const rawPresent = Boolean(record?.detector_card_present ?? record?.card_present);
      const sharpness = record?.sharpness;
      const sharpEnough = sharpness === null || sharpness === undefined || Number(sharpness) >= threshold;
      const quality = cornerQuality(record);
      const qualityEnough = quality === null || quality >= minCornerQuality;
      const score = record?.best_score !== null && record?.best_score !== undefined ? Number(record.best_score) : 0;
      const scoreEnough = score >= minScore;
      const present = rawPresent && sharpEnough && qualityEnough && scoreEnough && Boolean(record?.best_card_id);
      return {{
        present,
        thresholded: rawPresent && !present,
        id: present ? (record.best_card_id || null) : null,
        score: present ? score : 0,
        sharpness: sharpness === null || sharpness === undefined ? null : Number(sharpness),
        cornerQuality: quality,
        sharpEnough,
        qualityEnough,
        scoreEnough,
      }};
    }}
    function rejectionReason(record) {{
      return record?.rejection_reason || record?.corner_quality?.reason || '';
    }}
    function frameState(index, mode = comparisonMode) {{
      const threshold = sharpnessThreshold();
      const minScore = scoreThreshold();
      const minCornerQuality = cornerQualityThreshold();
      const records = data.labels.map(label => data.recordsByRun[label]?.[index]);
      const effective = records.map(record => effectiveRecord(record, threshold, minScore, minCornerQuality));
      const comparisonIds = records.map((record, recordIndex) => compareKeyForRecord(record, effective[recordIndex], mode));
      const distinct = new Set(comparisonIds);
      const disagrees = distinct.size > 1;
      const scores = effective.map(item => item.score);
      const sharpnesses = effective.map(item => item.sharpness).filter(value => value !== null && Number.isFinite(value));
      const sharpnessSpan = sharpnesses.length ? Math.max(...sharpnesses) - Math.min(...sharpnesses) : 0;
      const agreementStrength = disagrees || !effective.every(item => item.id)
        ? -1
        : Math.min(...scores);
      const disagreementStrength = disagrees
        ? distinct.size + scores.reduce((sum, value) => sum + value, 0) + sharpnessSpan
        : -1;
      return {{ records, effective, comparisonIds, disagrees, agreementStrength, disagreementStrength }};
    }}
    function recomputeIndexes() {{
      const disagreements = [];
      let strongestAgreementIndex = null;
      let strongestAgreementScore = -1;
      let strongestDisagreementIndex = null;
      let strongestDisagreementScore = -1;
      for (let index = 0; index < data.frameCount; index += 1) {{
        const state = frameState(index);
        if (state.disagrees) {{
          disagreements.push(index);
          if (state.disagreementStrength > strongestDisagreementScore) {{
            strongestDisagreementScore = state.disagreementStrength;
            strongestDisagreementIndex = index;
          }}
        }} else if (state.agreementStrength > strongestAgreementScore) {{
          strongestAgreementScore = state.agreementStrength;
          strongestAgreementIndex = index;
        }}
      }}
      indexCache = {{ disagreements, strongestAgreement: strongestAgreementIndex, strongestDisagreement: strongestDisagreementIndex }};
      agreementStats.textContent = `${{data.frameCount - disagreements.length}} agree, ${{disagreements.length}} disagree`;
      prevDisagreement.disabled = disagreements.length === 0;
      nextDisagreement.disabled = disagreements.length === 0;
      strongestAgreement.disabled = strongestAgreementIndex === null;
      strongestDisagreement.disabled = strongestDisagreementIndex === null;
    }}
    function jumpTo(index) {{
      if (index === null || index === undefined) return;
      frame.value = String(index + 1);
      render();
    }}
    function jumpDisagreement(direction) {{
      const current = Number(frame.value) - 1;
      const indexes = indexCache.disagreements;
      if (!indexes.length) return;
      const next = direction > 0
        ? indexes.find(index => index > current) ?? indexes[0]
        : [...indexes].reverse().find(index => index < current) ?? indexes[indexes.length - 1];
      jumpTo(next);
    }}
    function render() {{
      const index = Number(frame.value) - 1;
      frameLabel.textContent = `Frame ${{index + 1}} / ${{data.frameCount}}`;
      const firstRecord = data.recordsByRun[data.labels[0]]?.[index];
      timeLabel.textContent = firstRecord ? `t=${{fmt(firstRecord.timestamp_sec, 2)}}s` : '';
      const state = frameState(index);
      runs.innerHTML = data.labels.map((label, labelIndex) => {{
        const record = data.recordsByRun[label]?.[index] || {{}};
        const effective = state.effective[labelIndex];
        const hits = record.top_k || [];
        const crop = record.crop_path ? `<img src="${{record.crop_path}}" alt="crop">` : '<div class="meta">No crop saved</div>';
        const rows = hits.map(hit => `<tr><td>${{hit.rank}}</td><td>${{escapeHtml(cardDisplay(hit.card_id))}}</td><td>${{fmt(hit.score)}}</td><td>${{escapeHtml(cardDisplay(hit.oracle_id))}}</td></tr>`).join('');
        const presentText = effective.present ? 'yes' : (effective.thresholded ? 'no (under threshold)' : 'no');
        const bestText = effective.id ? cardDisplay(effective.id) : 'no detection';
        const bestScore = effective.id ? fmt(effective.score) : '';
        const quality = cornerQuality(record);
        const reason = rejectionReason(record);
        return `<section class="run ${{state.disagrees ? 'disagree' : ''}}">
          <h2>${{escapeHtml(label)}}</h2>
          <div class="status">
            <span class="${{effective.thresholded ? 'thresholded' : ''}}">present: ${{presentText}}</span>
            <span>sharpness: ${{fmt(record.sharpness, 4)}}</span>
            <span>corner quality: ${{quality === null ? '—' : fmt(quality, 2)}} ${{escapeHtml(reason)}}</span>
            <span>presence: ${{fmt(record.presence, 4)}}</span>
            <span>orientation: ${{escapeHtml(record.orientation || '')}}</span>
          </div>
          <div class="best">${{escapeHtml(bestText)}} ${{bestScore}}</div>
          <div class="media">
            <div>${{record.overlay_path ? `<img src="${{record.overlay_path}}" alt="overlay">` : ''}}</div>
            <div>${{crop}}</div>
          </div>
          <table><thead><tr><th>#</th><th>card</th><th>score</th><th>oracle</th></tr></thead><tbody>${{rows}}</tbody></table>
          <div class="meta">${{link(record.frame_path, 'frame')}} ${{link(record.overlay_path, 'overlay')}} ${{link(record.crop_path, 'crop')}}</div>
          ${{record.error ? `<p class="meta">error: ${{escapeHtml(record.error)}}</p>` : ''}}
        </section>`;
      }}).join('');
    }}
    function renderFrameCell(record, effective, comparisonKey) {{
      if (!record) return '<span class="thresholded">missing record</span>';
      const quality = cornerQuality(record);
      const rawPresent = Boolean(record?.detector_card_present ?? record?.card_present);
      let noMatchReason = 'no match';
      if (!rawPresent) noMatchReason = 'no corners';
      else if (quality !== null && quality < cornerQualityThreshold()) noMatchReason = rejectionReason(record) || 'corner quality';
      else if (record.sharpness !== null && record.sharpness !== undefined && Number(record.sharpness) < sharpnessThreshold()) noMatchReason = 'sharpness';
      else if (effective.thresholded) noMatchReason = 'under threshold';
      if (!effective.present) {{
        return `<span class="thresholded">${{escapeHtml(noMatchReason)}}</span><br><span class="meta">${{escapeHtml(missText(record, effective))}}</span>`;
      }}
      const displayId = comparisonMode === 'oracle' ? comparisonKey : effective.id;
      return `<div class="comparison-card">${{cardTitle(displayId)}}<span class="meta">print ${{escapeHtml(cardDisplay(effective.id))}}<br>score ${{fmt(effective.score)}} sharp ${{fmt(record.sharpness, 4)}} quality ${{quality === null ? '—' : fmt(quality, 2)}} ${{escapeHtml(record.orientation || '')}}</span></div>`;
    }}
    function pointsFor(records, getter, maxValue) {{
      const width = Math.max(1, records.length - 1);
      return records.map((record, index) => {{
        const raw = getter(record);
        const value = raw === null || raw === undefined || !Number.isFinite(Number(raw)) ? 0 : Math.max(0, Math.min(maxValue, Number(raw)));
        const x = 10 + (index / width) * 580;
        const y = 150 - (value / maxValue) * 130;
        return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
      }}).join(' ');
    }}
    function thresholdY(value, maxValue) {{
      return 150 - (Math.max(0, Math.min(maxValue, Number(value) || 0)) / maxValue) * 130;
    }}
    function buildMetricCharts() {{
      const panels = data.labels.map(label => {{
        const records = data.recordsByRun[label] || [];
        const sharpMax = Math.max(0.08, sharpnessThreshold() * 1.4, ...records.map(record => Number(record.sharpness) || 0));
        const scoreMax = 1;
        const sharpPoints = pointsFor(records, record => record.sharpness, sharpMax);
        const qualityPoints = pointsFor(records, record => cornerQuality(record), 1);
        const scorePoints = pointsFor(records, record => record.best_score, scoreMax);
        const sharpY = thresholdY(sharpnessThreshold(), sharpMax);
        const qualityY = thresholdY(cornerQualityThreshold(), 1);
        const scoreY = thresholdY(scoreThreshold(), scoreMax);
        return `<div class="chart-panel">
          <h3>${{escapeHtml(label)}}</h3>
          <svg class="metric-svg" viewBox="0 0 600 170" preserveAspectRatio="none" role="img" aria-label="${{escapeHtml(label)}} metrics over frames">
            <line x1="10" y1="150" x2="590" y2="150" stroke="#d8d1c3" stroke-width="1" />
            <line x1="10" y1="20" x2="590" y2="20" stroke="#ece7dc" stroke-width="1" />
            <line x1="10" y1="${{sharpY.toFixed(1)}}" x2="590" y2="${{sharpY.toFixed(1)}}" stroke="#2563eb" stroke-width="1" stroke-dasharray="4 4" />
            <line x1="10" y1="${{qualityY.toFixed(1)}}" x2="590" y2="${{qualityY.toFixed(1)}}" stroke="#0f766e" stroke-width="1" stroke-dasharray="4 4" />
            <line x1="10" y1="${{scoreY.toFixed(1)}}" x2="590" y2="${{scoreY.toFixed(1)}}" stroke="#b45309" stroke-width="1" stroke-dasharray="4 4" />
            <polyline points="${{sharpPoints}}" fill="none" stroke="#2563eb" stroke-width="1.6" vector-effect="non-scaling-stroke" />
            <polyline points="${{qualityPoints}}" fill="none" stroke="#0f766e" stroke-width="1.6" vector-effect="non-scaling-stroke" />
            <polyline points="${{scorePoints}}" fill="none" stroke="#b45309" stroke-width="1.6" vector-effect="non-scaling-stroke" />
          </svg>
          <div class="chart-legend">
            <span><span class="legend-swatch" style="background:#2563eb"></span>sharpness, max ${{fmt(sharpMax, 3)}}</span>
            <span><span class="legend-swatch" style="background:#0f766e"></span>corner quality</span>
            <span><span class="legend-swatch" style="background:#b45309"></span>top score</span>
            <span>dashed lines are current thresholds</span>
          </div>
        </div>`;
      }}).join('');
      metricCharts.innerHTML = `<h2>Metric timelines</h2><div class="meta">Sharpness, corner quality, and top-match score by frame. Raise or lower thresholds above to see which frames cross each limit.</div><div class="chart-grid">${{panels}}</div>`;
    }}
    function buildFrameRows() {{
      const rows = [];
      let hiddenAgreeCount = 0;
      for (let index = 0; index < data.frameCount; index += 1) {{
        const state = frameState(index, comparisonMode);
        const firstRecord = state.records[0] || {{}};
        const isAgreement = !state.disagrees;
        if (condenseAgreements && isAgreement) {{
          if (hiddenAgreeCount === 0) {{
            rows.push({{ kind: 'frame', index, state }});
          }} else if (rows[rows.length - 1]?.kind === 'hidden') {{
            rows[rows.length - 1].count += 1;
          }} else {{
            rows.push({{ kind: 'hidden', count: 1 }});
          }}
          hiddenAgreeCount += 1;
          continue;
        }}
        hiddenAgreeCount = 0;
        rows.push({{ kind: 'frame', index, state, firstRecord }});
      }}
      return rows;
    }}
    function buildFrameComparison() {{
      const allStates = Array.from({{ length: data.frameCount }}, (_, index) => frameState(index, comparisonMode));
      const disagreements = allStates.filter(state => state.disagrees).length;
      const tableRows = buildFrameRows().map(row => {{
        if (row.kind === 'hidden') {{
          return `<tr class="condensed"><td colspan="${{3 + data.labels.length}}">${{row.count}} agreeing frame${{row.count === 1 ? '' : 's'}} condensed</td></tr>`;
        }}
        const index = row.index;
        const state = row.state;
        const firstRecord = state.records[0] || {{}};
        const cells = data.labels.map((label, labelIndex) => `<td>${{renderFrameCell(state.records[labelIndex], state.effective[labelIndex], state.comparisonIds[labelIndex])}}</td>`).join('');
        const comparisonLabels = state.comparisonIds.map(id => id === 'NO_MATCH' ? id : cardDisplay(id)).join(' | ');
        return `<tr class="${{state.disagrees ? 'disagree' : 'agree'}}"><td><button type="button" data-frame="${{index + 1}}">${{index + 1}}</button></td><td>${{fmt(firstRecord.timestamp_sec, 2)}}s</td><td><span class="tag">${{state.disagrees ? 'disagree' : 'agree'}}</span><br>${{escapeHtml(comparisonLabels)}}</td>${{cells}}</tr>`;
      }}).join('');
      frameComparison.innerHTML = `<h2>Frame-by-frame comparison</h2>
        <div class="comparison-controls">
          <button type="button" data-comparison-mode="oracle" class="${{comparisonMode === 'oracle' ? 'active' : ''}}">Compare oracle / secondary ID</button>
          <button type="button" data-comparison-mode="card" class="${{comparisonMode === 'card' ? 'active' : ''}}">Compare exact card ID</button>
          <label class="threshold"><input id="condenseAgreements" type="checkbox" ${{condenseAgreements ? 'checked' : ''}}> condense agreeing rows</label>
        </div>
        <div class="meta">${{data.frameCount - disagreements}} agreeing frames, ${{disagreements}} disagreeing frames after sharpness >= ${{fmt(sharpnessThreshold(), 3)}}, corner quality >= ${{fmt(cornerQualityThreshold(), 2)}}, and score >= ${{fmt(scoreThreshold(), 2)}} thresholds.</div>
        <div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th>frame</th><th>time</th><th>comparison</th>${{data.labels.map(label => `<th>${{escapeHtml(label)}}</th>`).join('')}}</tr></thead><tbody>${{tableRows}}</tbody></table></div>`;
      frameComparison.querySelectorAll('button[data-frame]').forEach(button => {{
        button.addEventListener('click', () => jumpToFrameNumber(button.getAttribute('data-frame')));
      }});
      frameComparison.querySelectorAll('button[data-comparison-mode]').forEach(button => {{
        button.addEventListener('click', () => {{
          comparisonMode = button.getAttribute('data-comparison-mode') || 'oracle';
          recomputeIndexes();
          buildFrameComparison();
          render();
        }});
      }});
      const checkbox = document.getElementById('condenseAgreements');
      checkbox?.addEventListener('change', () => {{
        condenseAgreements = checkbox.checked;
        buildFrameComparison();
      }});
    }}
    function buildCardSummary() {{
      const labels = data.labels;
      const recordsByRun = data.recordsByRun || {{}};
      const threshold = sharpnessThreshold();
      const minScore = scoreThreshold();
      const minCornerQuality = cornerQualityThreshold();
      const groupedRowsByRun = {{}};
      const rawRowsByRun = {{}};
      function rowFromRecord(record) {{
        return {{
          card_id: record.best_card_id,
          oracle_id: recordOracleId(record),
          first_seen_frame: record.frame_index,
          first_seen_timestamp_sec: record.timestamp_sec,
          max_score: record.best_score,
          best_orientation: record.orientation,
          record,
        }};
      }}
      for (const label of labels) {{
        const grouped = new Map();
        const rawGrouped = new Map();
        for (const record of recordsByRun[label] || []) {{
          if (!record?.best_card_id) continue;
          const rawRow = rowFromRecord(record);
          const rawKey = cardKey(rawRow);
          if (rawKey && (!rawGrouped.has(rawKey) || rowTime(rawRow) < rowTime(rawGrouped.get(rawKey)))) rawGrouped.set(rawKey, rawRow);
          const effective = effectiveRecord(record, threshold, minScore, minCornerQuality);
          if (!effective.present) continue;
          const row = rowFromRecord(record);
          const key = cardKey(row);
          if (!key) continue;
          const existing = grouped.get(key);
          if (!existing || rowTime(row) < rowTime(existing)) grouped.set(key, row);
        }}
        groupedRowsByRun[label] = [...grouped.values()].sort((a, b) => rowFrame(a) - rowFrame(b));
        rawRowsByRun[label] = rawGrouped;
      }}
      const keyToRuns = new Map();
      for (const label of labels) {{
        for (const row of groupedRowsByRun[label] || []) {{
          const key = cardKey(row);
          if (!key) continue;
          if (!keyToRuns.has(key)) keyToRuns.set(key, new Map());
          const existing = keyToRuns.get(key).get(label);
          if (!existing || rowTime(row) < rowTime(existing)) keyToRuns.get(key).set(label, row);
        }}
      }}
      const sharedCount = [...keyToRuns.values()].filter(runMap => labels.every(label => runMap.has(label))).length;
      const runTotals = labels.map(label => (groupedRowsByRun[label] || []).length);
      const panels = labels.map(label => {{
        const rows = groupedRowsByRun[label] || [];
        const pills = rows.map(row => {{
          const key = cardKey(row);
          const shared = keyToRuns.get(key)?.size === labels.length;
          const cls = shared ? 'shared' : 'only';
          const tag = shared ? 'shared' : 'only';
          return `<button class="card-pill ${{cls}}" type="button" data-frame="${{escapeHtml(row.first_seen_frame)}}"><span class="tag">${{tag}}</span><span>${{escapeHtml(cardDisplay(key))}}</span><span>${{fmt(row.first_seen_timestamp_sec, 1)}}s ${{escapeHtml(row.best_orientation || '')}}</span></button>`;
        }}).join('');
        return `<div class="summary-panel"><h3>${{escapeHtml(label)}} <span class="meta">${{rows.length}} cards</span></h3><div class="card-list">${{pills}}</div></div>`;
      }}).join('');
      const chronological = [...keyToRuns.entries()].sort((a, b) => {{
        const minA = Math.min(...[...a[1].values()].map(rowTime));
        const minB = Math.min(...[...b[1].values()].map(rowTime));
        return minA - minB;
      }});
      const tableRows = chronological.map(([key, runMap]) => {{
        const shared = labels.every(label => runMap.has(label));
        const cells = labels.map(label => {{
          const row = runMap.get(label);
          if (!row) {{
            const rawRow = rawRowsByRun[label]?.get(key);
            if (!rawRow) return `<td class="thresholded">missing<br><span class="meta">no matching candidate for current ${{escapeHtml(groupLabel())}}</span></td>`;
            const effective = effectiveRecord(rawRow.record, threshold, minScore, minCornerQuality);
            return `<td class="thresholded">filtered<br><span class="meta">${{escapeHtml(missText(rawRow.record, effective))}}</span><br><button type="button" data-frame="${{escapeHtml(rawRow.first_seen_frame)}}">frame ${{escapeHtml(rawRow.first_seen_frame)}}</button></td>`;
          }}
          return `<td><button type="button" data-frame="${{escapeHtml(row.first_seen_frame)}}">frame ${{escapeHtml(row.first_seen_frame)}}</button> <span class="meta">${{fmt(row.first_seen_timestamp_sec, 1)}}s score ${{fmt(row.max_score)}} ${{escapeHtml(row.best_orientation || '')}}</span><br>${{escapeHtml(cardDisplay(row.card_id))}}</td>`;
        }}).join('');
        return `<tr class="${{shared ? 'shared' : 'mismatch'}}"><td><span class="tag">${{shared ? 'shared' : 'mismatch'}}</span><br>${{escapeHtml(cardDisplay(key))}}</td>${{cells}}</tr>`;
      }}).join('');
      cardSummary.innerHTML = `<h2>Detected cards summary</h2>
        <div class="summary-tabs"><button type="button" data-summary-mode="oracle" class="${{summaryMode === 'oracle' ? 'active' : ''}}">Oracle / secondary ID</button><button type="button" data-summary-mode="card" class="${{summaryMode === 'card' ? 'active' : ''}}">Exact card ID</button></div>
        <div class="meta">Grouped by ${{groupLabel()}}. Filtering summary and frame agreement at sharpness >= ${{fmt(threshold, 3)}}, corner quality >= ${{fmt(minCornerQuality, 2)}}, and score >= ${{fmt(minScore, 2)}}. ${{sharedCount}} shared identities across all runs. Totals: ${{labels.map((label, index) => `${{label}}=${{runTotals[index]}}`).join(', ')}}</div>
        <div class="summary-grid">${{panels}}</div>
        <div class="summary-table-wrap"><table class="summary-table"><thead><tr><th>card identity</th>${{labels.map(label => `<th>${{escapeHtml(label)}}</th>`).join('')}}</tr></thead><tbody>${{tableRows}}</tbody></table></div>`;
      cardSummary.querySelectorAll('button[data-summary-mode]').forEach(button => {{
        button.addEventListener('click', () => {{
          summaryMode = button.getAttribute('data-summary-mode') || 'oracle';
          buildCardSummary();
        }});
      }});
      cardSummary.querySelectorAll('button[data-frame]').forEach(button => {{
        button.addEventListener('click', () => jumpToFrameNumber(button.getAttribute('data-frame')));
      }});
    }}
    frame.addEventListener('input', render);
    thresholdInput.addEventListener('input', () => {{ recomputeIndexes(); buildMetricCharts(); buildFrameComparison(); buildCardSummary(); render(); }});
    cornerQualityThresholdInput.addEventListener('input', () => {{ recomputeIndexes(); buildMetricCharts(); buildFrameComparison(); buildCardSummary(); render(); }});
    scoreThresholdInput.addEventListener('input', () => {{ recomputeIndexes(); buildMetricCharts(); buildFrameComparison(); buildCardSummary(); render(); }});
    prevDisagreement.addEventListener('click', () => jumpDisagreement(-1));
    nextDisagreement.addEventListener('click', () => jumpDisagreement(1));
    strongestAgreement.addEventListener('click', () => jumpTo(indexCache.strongestAgreement));
    strongestDisagreement.addEventListener('click', () => jumpTo(indexCache.strongestDisagreement));
    recomputeIndexes();
    buildMetricCharts();
    buildFrameComparison();
    buildCardSummary();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Output directory containing manifest.json")
    parser.add_argument("--output", type=Path, help="HTML output path")
    parser.add_argument("--no-card-names", action="store_true", help="Do not load cached Scryfall bulk names")
    args = parser.parse_args()
    write_viewer(args.run_dir, args.output, card_names=not args.no_card_names)


if __name__ == "__main__":
    main()
