(function () {
  var BROADCAST_TARGET_KEY = 'nifty_cv_broadcast_target';
  var defaultBroadcastTarget = window.location.origin;
  var broadcastTarget = localStorage.getItem(BROADCAST_TARGET_KEY) || defaultBroadcastTarget;
  var socket = window.io.connect(broadcastTarget, { transports: ['websocket'] });
  var video = document.getElementById('video');
  var overlay = document.getElementById('overlay');
  var overlayCtx = overlay.getContext('2d');
  var captureCanvas = document.getElementById('capture-canvas');
  var captureCtx = captureCanvas.getContext('2d');
  var stream = null;
  var scanTimer = null;
  var scanInFlight = false;
  var lastScanStartedAt = 0;
  var embeddingBuffer = [];
  var matchWindow = [];
  var scryfallCache = new Map();
  var prefetchedImages = new Set();
  var lastResult = null;
  var lastFpsTimestamp = null;
  var fpsEma = null;

  var controls = {
    cameraSelect: document.getElementById('camera-select'),
    refreshCameras: document.getElementById('refresh-cameras'),
    startCamera: document.getElementById('start-camera'),
    stopCamera: document.getElementById('stop-camera'),
    scanOnce: document.getElementById('scan-once'),
    broadcastCurrent: document.getElementById('broadcast-current'),
    broadcastTarget: document.getElementById('broadcast-target'),
    broadcastTargetStatus: document.getElementById('broadcast-target-status'),
    lastBroadcastStatus: document.getElementById('last-broadcast-status'),
    applyBroadcastTarget: document.getElementById('apply-broadcast-target'),
    resetBroadcastTarget: document.getElementById('reset-broadcast-target'),
    autoScan: document.getElementById('auto-scan'),
    broadcastConfirmed: document.getElementById('broadcast-confirmed'),
    prefetchImages: document.getElementById('prefetch-images'),
    rotationInvariant: document.getElementById('rotation-invariant'),
    bucketBySecondary: document.getElementById('bucket-by-secondary'),
    clearBuckets: document.getElementById('clear-buckets')
  };

  var sliders = {
    scanInterval: ['scan-interval', 'scan-interval-value', function (v) { return Number(v) === 0 ? 'Free-running' : v + ' ms'; }],
    jpegQuality: ['jpeg-quality', 'jpeg-quality-value', function (v) { return Number(v).toFixed(2); }],
    minSharpness: ['min-sharpness', 'min-sharpness-value', function (v) { return Number(v).toFixed(3); }],
    embeddingBuffer: ['embedding-buffer', 'embedding-buffer-value', function (v) { return v; }],
    priorSimilarity: ['prior-similarity', 'prior-similarity-value', function (v) { return Number(v).toFixed(2); }],
    bucketWindow: ['bucket-window', 'bucket-window-value', function (v) { return v; }],
    matchThreshold: ['match-threshold', 'match-threshold-value', function (v) { return Number(v).toFixed(2); }],
    confirmCount: ['confirm-count', 'confirm-count-value', function (v) { return v; }]
  };

  function el(id) { return document.getElementById(id); }
  function value(name) { return Number(el(sliders[name][0]).value); }
  function setStatus(text) { el('camera-status').textContent = text; }
  function formatNumber(num, digits) { return Number.isFinite(num) ? num.toFixed(digits) : '-'; }

  function updateFps() {
    var now = performance.now();
    if (lastFpsTimestamp === null) {
      lastFpsTimestamp = now;
      el('fps-overlay').textContent = 'FPS --';
      return;
    }
    var delta = now - lastFpsTimestamp;
    lastFpsTimestamp = now;
    if (!Number.isFinite(delta) || delta <= 0) return;
    var fps = 1000 / delta;
    fpsEma = fpsEma === null ? fps : fpsEma * 0.8 + fps * 0.2;
    el('fps-overlay').textContent = 'FPS ' + fpsEma.toFixed(1);
  }

  function sharpnessMax() {
    return Number(el('min-sharpness').max) || 0.08;
  }

  function updateSharpnessMeter(currentSharpness) {
    var current = Number(currentSharpness);
    var threshold = value('minSharpness');
    var max = sharpnessMax();
    var currentPct = Number.isFinite(current) ? Math.min(100, Math.max(0, (current / max) * 100)) : 0;
    var thresholdPct = Math.min(100, Math.max(0, (threshold / max) * 100));
    el('sharpness-fill').style.width = currentPct + '%';
    el('sharpness-threshold').style.left = thresholdPct + '%';
    el('sharpness-current').textContent = Number.isFinite(current) ? current.toFixed(3) : '-';
  }

  function updateUnitMeter(kind, currentValue, thresholdValue, minValue, maxValue) {
    var current = Number(currentValue);
    var threshold = Number(thresholdValue);
    var min = Number(minValue);
    var max = Number(maxValue);
    var span = max - min;
    var currentPct = Number.isFinite(current) && span > 0
      ? Math.min(100, Math.max(0, ((current - min) / span) * 100))
      : 0;
    var thresholdPct = Number.isFinite(threshold) && span > 0
      ? Math.min(100, Math.max(0, ((threshold - min) / span) * 100))
      : 0;
    el(kind + '-fill').style.width = currentPct + '%';
    el(kind + '-threshold').style.left = thresholdPct + '%';
    el(kind + '-current').textContent = Number.isFinite(current) ? current.toFixed(3) : '-';
  }

  function updateScoreMeter(currentScore) {
    updateUnitMeter('score', currentScore, value('matchThreshold'), 0, 1);
  }

  function updatePriorMeter(currentSimilarity) {
    updateUnitMeter('prior', currentSimilarity, value('priorSimilarity'), 0, 1);
  }

  function updateConfirmMeter(bestCount) {
    var count = Math.max(0, Number(bestCount) || 0);
    var required = Math.max(1, value('confirmCount'));
    var capacity = Math.max(required, value('bucketWindow'));
    var pct = Math.min(100, (count / capacity) * 100);
    var thresholdPct = Math.min(100, (required / capacity) * 100);
    el('confirm-fill').style.width = pct + '%';
    el('confirm-threshold').style.left = thresholdPct + '%';
    el('confirm-current').textContent = Math.min(count, capacity) + ' / ' + capacity;
  }

  function normalizeTarget(target) {
    var raw = (target || '').trim();
    if (!raw) return defaultBroadcastTarget;
    return new URL(raw, window.location.href).origin;
  }

  function updateBroadcastTargetStatus() {
    controls.broadcastTarget.value = broadcastTarget;
    controls.broadcastTargetStatus.textContent = broadcastTarget === defaultBroadcastTarget
      ? 'Broadcasting to this server.'
      : 'Broadcasting to ' + broadcastTarget + '.';
  }

  function setBroadcastTarget(target) {
    broadcastTarget = normalizeTarget(target);
    if (broadcastTarget === defaultBroadcastTarget) {
      localStorage.removeItem(BROADCAST_TARGET_KEY);
    } else {
      localStorage.setItem(BROADCAST_TARGET_KEY, broadcastTarget);
    }
    socket = window.io.connect(broadcastTarget, { transports: ['websocket'] });
    updateBroadcastTargetStatus();
  }

  function updateSliderLabels() {
    Object.keys(sliders).forEach(function (name) {
      var data = sliders[name];
      el(data[1]).textContent = data[2](el(data[0]).value);
    });
  }

  function resizeOverlay() {
    var rect = video.getBoundingClientRect();
    var scale = window.devicePixelRatio || 1;
    overlay.width = Math.max(1, Math.round(rect.width * scale));
    overlay.height = Math.max(1, Math.round(rect.height * scale));
    overlayCtx.setTransform(scale, 0, 0, scale, 0, 0);
  }

  function drawCorners(corners) {
    resizeOverlay();
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    if (!corners || !corners.length) return;
    var rect = video.getBoundingClientRect();
    var videoAspect = video.videoWidth / video.videoHeight;
    var frameAspect = rect.width / rect.height;
    var drawWidth = rect.width;
    var drawHeight = rect.height;
    var offsetX = 0;
    var offsetY = 0;
    if (videoAspect > frameAspect) {
      drawHeight = rect.width / videoAspect;
      offsetY = (rect.height - drawHeight) / 2;
    } else {
      drawWidth = rect.height * videoAspect;
      offsetX = (rect.width - drawWidth) / 2;
    }

    overlayCtx.strokeStyle = '#39d1a5';
    overlayCtx.lineWidth = 3;
    overlayCtx.beginPath();
    corners.forEach(function (point, index) {
      var x = offsetX + point[0] * drawWidth;
      var y = offsetY + point[1] * drawHeight;
      if (index === 0) overlayCtx.moveTo(x, y);
      else overlayCtx.lineTo(x, y);
    });
    overlayCtx.closePath();
    overlayCtx.stroke();
  }

  async function listCameras() {
    var devices = await navigator.mediaDevices.enumerateDevices();
    var cameras = devices.filter(function (device) { return device.kind === 'videoinput'; });
    controls.cameraSelect.innerHTML = '';
    cameras.forEach(function (camera, index) {
      var option = document.createElement('option');
      option.value = camera.deviceId;
      option.textContent = camera.label || 'Camera ' + (index + 1);
      controls.cameraSelect.appendChild(option);
    });
    if (!cameras.length) {
      var option = document.createElement('option');
      option.textContent = 'No cameras found';
      controls.cameraSelect.appendChild(option);
    }
  }

  async function startCamera() {
    stopCamera();
    var deviceId = controls.cameraSelect.value;
    var constraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'environment' },
      audio: false
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    await listCameras();
    controls.startCamera.disabled = true;
    controls.stopCamera.disabled = false;
    setStatus('Camera running');
    resizeOverlay();
    scheduleScanning();
  }

  function stopCamera() {
    if (scanTimer) {
      clearTimeout(scanTimer);
      scanTimer = null;
    }
    if (stream) {
      stream.getTracks().forEach(function (track) { track.stop(); });
      stream = null;
    }
    video.srcObject = null;
    lastFpsTimestamp = null;
    fpsEma = null;
    el('fps-overlay').textContent = 'FPS --';
    controls.startCamera.disabled = false;
    controls.stopCamera.disabled = true;
    setStatus('Camera idle');
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function scheduleScanning() {
    if (scanTimer) {
      clearTimeout(scanTimer);
      scanTimer = null;
    }
    if (!controls.autoScan.checked || !stream) return;
    scheduleNextScan();
  }

  function scheduleNextScan() {
    if (scanTimer) {
      clearTimeout(scanTimer);
      scanTimer = null;
    }
    if (!controls.autoScan.checked || !stream) return;
    var interval = value('scanInterval');
    var elapsedSinceStart = performance.now() - lastScanStartedAt;
    var delay = lastScanStartedAt ? Math.max(0, interval - elapsedSinceStart) : 0;
    scanTimer = setTimeout(function () {
      scanTimer = null;
      scanFrame(true);
    }, delay);
  }

  function captureFrame() {
    if (!video.videoWidth || !video.videoHeight) return null;
    var maxWidth = 960;
    var scale = Math.min(1, maxWidth / video.videoWidth);
    captureCanvas.width = Math.round(video.videoWidth * scale);
    captureCanvas.height = Math.round(video.videoHeight * scale);
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    return captureCanvas.toDataURL('image/jpeg', value('jpegQuality')).split(',')[1];
  }

  async function scanFrame(fromAuto) {
    if (scanInFlight) {
      if (fromAuto) scheduleNextScan();
      return;
    }
    if (!stream) return;
    var base64 = captureFrame();
    if (!base64) {
      if (fromAuto) scheduleNextScan();
      return;
    }
    scanInFlight = true;
    lastScanStartedAt = performance.now();
    setStatus('Scanning frame');
    try {
      var response = await fetch('/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          _base64: base64,
          top_k: 1,
          prior_embeddings: embeddingBuffer.slice(-value('embeddingBuffer')),
          min_sharpness: value('minSharpness'),
          min_prior_similarity: value('priorSimilarity'),
          rotation_invariant: controls.rotationInvariant.checked,
          broadcast: false
        })
      });
      if (!response.ok) throw new Error(await response.text());
      handleResult(await response.json());
      updateFps();
    } catch (err) {
      setDecision('rejected', 'Error');
      el('candidate-note').textContent = err.message || String(err);
      setStatus('Scan failed');
    } finally {
      scanInFlight = false;
      if (fromAuto) scheduleNextScan();
    }
  }

  function handleResult(result) {
    lastResult = result;
    updateMetrics(result);

    if (!result.card_present) {
      drawCorners(null);
      clearCandidatePreview();
      drainBuckets();
      setDecision('waiting', 'No card');
      el('candidate-note').textContent = 'Detector gates rejected this frame.';
      setStatus('No card accepted');
      return;
    }

    drawCorners(result.corners);

    if (result.confidence < value('matchThreshold')) {
      setDecision('rejected', 'Weak score');
      el('candidate-note').textContent = 'Score ' + formatNumber(result.confidence, 3) + ' is below threshold.';
      drainBuckets();
      setStatus('Below score threshold');
      return;
    }

    if (result.embedding && value('embeddingBuffer') > 0) {
      embeddingBuffer.push(result.embedding);
      while (embeddingBuffer.length > value('embeddingBuffer')) embeddingBuffer.shift();
    }

    var decision = addBucket(result, true);
    if (decision.confirmed) {
      setDecision('confirmed', 'Confirmed');
      el('candidate-note').textContent = 'Confirmed from ' + decision.count + ' frames.';
      controls.broadcastCurrent.disabled = false;
      maybeBroadcast(result);
    } else {
      setDecision('waiting', 'Collecting');
      el('candidate-note').textContent = decision.count + ' of ' + value('confirmCount') + ' frames collected.';
    }
    setStatus('Frame processed');
  }

  function updateMetrics(result) {
    updateCandidateLabel(result.card_id);
    el('metric-match').textContent = formatNumber(result.confidence, 3);
    updateScoreMeter(result.confidence);
    el('metric-sharpness').textContent = formatNumber(result.sharpness, 3);
    updateSharpnessMeter(result.sharpness);
    updatePriorMeter(result.prior_similarity);
    el('metric-total').textContent = result._timing ? Math.round(result._timing.total_ms) + ' ms' : '-';
    if (result.crop_jpeg) {
      el('crop-preview').src = 'data:image/jpeg;base64,' + result.crop_jpeg;
    }
  }

  function clearCandidatePreview() {
    el('candidate-id').textContent = 'No candidate';
    el('metric-match').textContent = '-';
    el('metric-total').textContent = '-';
    var preview = el('crop-preview');
    preview.removeAttribute('src');
  }

  function prefetchCardImage(src) {
    if (!controls.prefetchImages.checked) return;
    if (!src || prefetchedImages.has(src)) return;
    prefetchedImages.add(src);

    var link = document.createElement('link');
    link.rel = 'prefetch';
    link.as = 'image';
    link.href = src;
    document.head.appendChild(link);

    var image = new Image();
    image.decoding = 'async';
    image.src = src;
  }

  function addBucket(result, accepted) {
    if (accepted) {
      prefetchCardImage(result.image_src);
      matchWindow.push({
        cardId: result.card_id,
        oracleId: secondaryIdFor(result) || cachedOracleIdFor(result.card_id),
        imageSrc: result.image_src,
        confidence: result.confidence,
        at: Date.now()
      });
      while (matchWindow.length > value('bucketWindow')) matchWindow.shift();
    }

    var buckets = summarizeBuckets();
    renderBuckets(buckets);
    var best = buckets[0] || { cardId: result.card_id, count: 0, avg: 0, imageSrc: result.image_src };
    updateConfirmMeter(best.count);
    return {
      confirmed: best.count >= value('confirmCount') && best.avg >= value('matchThreshold'),
      cardId: best.cardId,
      imageSrc: best.imageSrc,
      count: best.count,
      avg: best.avg
    };
  }

  function drainBuckets() {
    if (matchWindow.length > 0) {
      matchWindow.shift();
    }
    var buckets = summarizeBuckets();
    renderBuckets(buckets);
    updateConfirmMeter((buckets[0] || { count: 0 }).count);
  }

  function summarizeBuckets() {
    var map = new Map();
    matchWindow.forEach(function (item) {
      var bucketKey = bucketKeyForItem(item);
      var bucket = map.get(bucketKey) || {
        bucketKey: bucketKey,
        cardId: item.cardId,
        oracleId: item.oracleId,
        imageSrc: item.imageSrc,
        count: 0,
        scoreSum: 0,
        max: 0
      };
      bucket.count += 1;
      bucket.scoreSum += item.confidence;
      if (item.confidence >= bucket.max) {
        bucket.max = item.confidence;
        bucket.cardId = item.cardId;
        bucket.oracleId = item.oracleId;
        bucket.imageSrc = item.imageSrc;
      }
      map.set(bucketKey, bucket);
    });
    return Array.from(map.values()).map(function (bucket) {
      bucket.avg = bucket.scoreSum / bucket.count;
      return bucket;
    }).sort(function (a, b) {
      return b.count - a.count || b.avg - a.avg;
    });
  }

  function renderBuckets(buckets) {
    var list = el('bucket-list');
    list.innerHTML = '';
    buckets.slice(0, 6).forEach(function (bucket) {
      var item = document.createElement('li');
      item.innerHTML = '<div class="bucket-list__top"><strong></strong><span></span></div><div class="bucket-list__meter"><i></i><b></b></div>';
      item.querySelector('strong').textContent = bucketLabel(bucket);
      item.querySelector('span').textContent = bucket.count + ' frames / ' + formatNumber(bucket.avg, 3);
      updateBucketRowMeter(item, bucket.count);
      list.appendChild(item);
      warmBucketLabel(bucket).then(function () {
        item.querySelector('strong').textContent = bucketLabel(bucket);
      });
    });
  }

  function updateBucketRowMeter(item, count) {
    var required = Math.max(1, value('confirmCount'));
    var capacity = Math.max(required, value('bucketWindow'));
    var fillPct = Math.min(100, (Math.max(0, Number(count) || 0) / capacity) * 100);
    var thresholdPct = Math.min(100, (required / capacity) * 100);
    item.querySelector('.bucket-list__meter i').style.width = fillPct + '%';
    item.querySelector('.bucket-list__meter b').style.left = thresholdPct + '%';
  }

  function secondaryIdFor(candidate) {
    var explicit = String(candidate.secondaryId || candidate.oracle_id || candidate.oracleId || '').trim();
    if (explicit) return explicit;
    var field = String(candidate.secondaryIdField || '').trim();
    return field ? String(candidate[field] || '').trim() : '';
  }

  function cachedOracleIdFor(cardId) {
    return String(scryfallCache.get(cardId)?.oracleId || '').trim();
  }

  function bucketKeyFor(candidate) {
    var cardId = String(candidate.card_id || '').trim();
    if (controls.bucketBySecondary.checked) {
      var secondaryId = secondaryIdFor(candidate);
      if (secondaryId) return 'secondary:' + secondaryId;
    }
    return 'card:' + cardId;
  }

  function bucketKeyForItem(item) {
    if (controls.bucketBySecondary.checked && item.oracleId) {
      return 'secondary:' + item.oracleId;
    }
    return 'card:' + item.cardId;
  }

  function bucketLabel(bucket) {
    var label = cardLabel(bucket.cardId);
    if (label) return label;
    return bucket.oracleId && controls.bucketBySecondary.checked
      ? bucket.oracleId + ' (oracle)'
      : bucket.cardId;
  }

  function cardLabel(cardId) {
    var cached = scryfallCache.get(cardId);
    if (cached && cached.name) {
      return cached.name + ' [' + String(cached.set || '').toUpperCase() + ']';
    }
    return '';
  }

  function updateCandidateLabel(cardId) {
    if (!cardId) {
      el('candidate-id').textContent = 'No candidate';
      return;
    }
    el('candidate-id').textContent = cardLabel(cardId) || cardId;
    el('candidate-note').textContent = cardLabel(cardId) ? 'Latest recognized card.' : 'Resolving ' + cardId + '.';
    warmCardMetadata(cardId).then(function () {
      if (lastResult && lastResult.card_id === cardId) {
        el('candidate-id').textContent = cardLabel(cardId) || cardId;
        if (cardLabel(cardId)) {
          el('candidate-note').textContent = 'Latest recognized card.';
        }
      }
    });
  }

  async function warmBucketLabel(bucket) {
    await warmCardMetadata(bucket.cardId);
    var cached = scryfallCache.get(bucket.cardId);
    if (cached?.oracleId && backfillOracleId(bucket.cardId, cached.oracleId)) {
      renderBuckets(summarizeBuckets());
    }
  }

  async function warmCardMetadata(cardId) {
    if (!cardId || scryfallCache.has(cardId)) return;
    scryfallCache.set(cardId, { loading: true });
    try {
      var response = await fetch('https://api.scryfall.com/cards/' + encodeURIComponent(cardId));
      if (!response.ok) throw new Error('Scryfall ' + response.status);
      var card = await response.json();
      scryfallCache.set(cardId, {
        name: card.name || cardId,
        set: card.set || card.set_name || '',
        oracleId: card.oracle_id || ''
      });
    } catch (err) {
      scryfallCache.set(cardId, { name: cardId, set: '', oracleId: '' });
    }
  }

  function backfillOracleId(cardId, oracleId) {
    var changed = false;
    matchWindow.forEach(function (item) {
      if (item.cardId === cardId && item.oracleId !== oracleId) {
        item.oracleId = oracleId;
        changed = true;
      }
    });
    return changed;
  }

  function maybeBroadcast(result) {
    if (!controls.broadcastConfirmed.checked || !result.image_src) return;
    broadcast(result);
  }

  function broadcast(result) {
    socket.emit('card_image', { auto: true, src: result.image_src });
    updateLastBroadcast(result.card_id);
    setStatus('Broadcast ' + result.card_id + ' to ' + broadcastTarget);
  }

  function updateLastBroadcast(cardId) {
    var sentAt = new Date();
    function render() {
      controls.lastBroadcastStatus.textContent = 'Last broadcast: '
        + (cardLabel(cardId) || cardId)
        + ' at '
        + sentAt.toLocaleTimeString();
    }
    render();
    warmCardMetadata(cardId).then(render);
  }

  function setDecision(state, label) {
    el('decision-strip').dataset.state = state;
    el('decision-state').textContent = label;
  }

  Object.keys(sliders).forEach(function (name) {
    el(sliders[name][0]).addEventListener('input', function () {
      updateSliderLabels();
      if (name === 'scanInterval') scheduleScanning();
      if (name === 'minSharpness') updateSharpnessMeter(lastResult ? lastResult.sharpness : null);
      if (name === 'matchThreshold') updateScoreMeter(lastResult ? lastResult.confidence : null);
      if (name === 'priorSimilarity') updatePriorMeter(lastResult ? lastResult.prior_similarity : null);
      if (name === 'confirmCount') updateConfirmMeter((summarizeBuckets()[0] || { count: 0 }).count);
      if (name === 'embeddingBuffer') {
        while (embeddingBuffer.length > value('embeddingBuffer')) embeddingBuffer.shift();
      }
      if (name === 'bucketWindow') {
        while (matchWindow.length > value('bucketWindow')) matchWindow.shift();
        var buckets = summarizeBuckets();
        renderBuckets(buckets);
        updateConfirmMeter((buckets[0] || { count: 0 }).count);
      }
    });
  });

  controls.bucketBySecondary.addEventListener('change', function () {
    renderBuckets(summarizeBuckets());
    setDecision('idle', 'Waiting');
    el('candidate-note').textContent = controls.bucketBySecondary.checked ? 'Bucketing by oracle ID.' : 'Bucketing by exact card ID.';
  });

  controls.refreshCameras.addEventListener('click', listCameras);
  controls.startCamera.addEventListener('click', function () { startCamera().catch(function (err) { setStatus(err.message); }); });
  controls.stopCamera.addEventListener('click', stopCamera);
  controls.scanOnce.addEventListener('click', function () { scanFrame(false); });
  controls.autoScan.addEventListener('change', scheduleScanning);
  controls.broadcastCurrent.addEventListener('click', function () { if (lastResult) broadcast(lastResult); });
  controls.applyBroadcastTarget.addEventListener('click', function () {
    try {
      setBroadcastTarget(controls.broadcastTarget.value);
    } catch (err) {
      controls.broadcastTargetStatus.textContent = 'Invalid target URL.';
    }
  });
  controls.resetBroadcastTarget.addEventListener('click', function () { setBroadcastTarget(defaultBroadcastTarget); });
  controls.clearBuckets.addEventListener('click', function () {
    matchWindow = [];
    embeddingBuffer = [];
    renderBuckets([]);
    updateConfirmMeter(0);
    setDecision('idle', 'Waiting');
    el('candidate-note').textContent = 'Buckets cleared.';
  });
  window.addEventListener('resize', resizeOverlay);

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('Camera API unavailable');
  } else {
    listCameras().catch(function () { setStatus('Camera permission needed'); });
  }
  updateBroadcastTargetStatus();
  updateSharpnessMeter(null);
  updateScoreMeter(null);
  updatePriorMeter(null);
  updateConfirmMeter(0);
  updateSliderLabels();
}());