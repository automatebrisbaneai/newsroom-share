/**
 * Voice-to-Text v1.0
 * Drop-in voice transcription with blur interim display.
 *
 * Usage:
 *   <script src="/shared/voice-to-text.js"></script>
 *   <script>
 *     VoiceToText.init({
 *       target:   'captionField',       // textarea ID to fill
 *       button:   'voiceBtn',           // mic button ID (optional — auto-creates if missing)
 *       interim:  'interim',            // interim display div ID (optional — auto-creates if missing)
 *       status:   'voiceStatus',        // status text element ID (optional)
 *       label:    'voiceLabel',         // label element ID (optional)
 *       cleanUrl: '/clean',             // POST endpoint for transcript cleanup (default: /clean)
 *       lang:     'en-AU',             // recognition language (default: en-AU)
 *     });
 *   </script>
 */

(function () {
  'use strict';

  // ── Inject CSS once ──────────────────────────────────────────────────────
  const STYLE_ID = 'vtt-styles';
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      .vtt-interim {
        display: none;
        width: 100%;
        margin-top: 8px;
        padding: 11px 13px;
        font-size: 0.95rem;
        font-family: inherit;
        border: 1.5px solid #dc2626;
        border-radius: 8px;
        background: white;
        color: #1a1a1a;
        min-height: 90px;
        max-height: 50vh;
        overflow-y: auto;
        line-height: 1.55;
        word-wrap: break-word;
      }
      .vtt-interim .vtt-blurred {
        filter: blur(5px);
        opacity: 0.55;
        color: #737373;
        user-select: none;
      }
      .vtt-interim .vtt-clear {
        filter: none;
        opacity: 1;
      }
      .vtt-interim .vtt-cleaned {
        color: #0f172a;
        font-weight: 500;
      }
      .vtt-btn {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: 1.5px solid #e8e6e3;
        background: white;
        color: #737373;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        -webkit-tap-highlight-color: transparent;
        transition: background 0.15s, border-color 0.15s, color 0.15s;
        flex-shrink: 0;
      }
      .vtt-btn:active { background: #f4f4f2; }
      .vtt-btn.vtt-recording {
        background: #fef2f2;
        border-color: #dc2626;
        color: #dc2626;
        animation: vttPulse 1.4s ease-in-out infinite;
      }
      .vtt-btn:disabled { opacity: 0.4; cursor: not-allowed; }
      .vtt-edit-hint {
        font-size: 0.72rem;
        color: #73182C;
        text-align: center;
        margin-top: 6px;
        opacity: 0;
        transition: opacity 0.3s ease;
      }
      .vtt-edit-hint.vtt-visible { opacity: 1; }
      .vtt-textarea-highlight {
        border-color: #73182C !important;
        box-shadow: 0 0 0 3px rgba(115, 24, 44, 0.1) !important;
        transition: border-color 0.3s, box-shadow 0.3s;
      }
      @keyframes vttPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.15); }
        50%       { box-shadow: 0 0 0 5px rgba(220,38,38,0.08); }
      }
      @keyframes vttChunkReveal {
        from { filter: blur(4px); opacity: 0.4; }
        to   { filter: none;      opacity: 1;   }
      }
      .vtt-chunk-new {
        animation: vttChunkReveal 0.4s ease forwards;
      }
      .vtt-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #73182C;
        margin-left: 5px;
        vertical-align: middle;
        animation: vttDotPulse 1.6s ease-in-out infinite;
      }
      @keyframes vttDotPulse {
        0%, 100% { opacity: 0.35; transform: scale(0.85); }
        50%      { opacity: 1;    transform: scale(1.1);  }
      }
    `;
    document.head.appendChild(style);
  }

  // ── SVGs ─────────────────────────────────────────────────────────────────
  const MIC_SVG  = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>';
  const STOP_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

  // ── Chunk timing ────────────────────────────────────────────────────────
  // Ship a progressive chunk to /clean when the speaker pauses for PAUSE_MS.
  // MAX_CHUNK_MS is a safety fallback so continuous talkers still see output.
  const PAUSE_MS       = 2500;
  const MAX_CHUNK_MS   = 120000;
  const TICK_MS        = 500;
  // Minimum finalised-text length before a pause will flush a chunk.
  // A thinking "um" with an accidental final-result flush is noise the
  // LLM mishandles — skip it until either the speaker keeps talking (so
  // the chunk grows) or MAX_CHUNK_MS fires (safety fallback).
  const MIN_CHUNK_CHARS = 40;

  // ── Meta-response detector ──────────────────────────────────────────────
  // Matches the same patterns the backend fallbacks catch — this is
  // defence-in-depth for when an old container is still running the
  // prior prompt and the server-side guard isn't live yet.
  const META_START = /^\s*(sure|certainly|okay|of course|please|here is|here's|understood|i (can|will|am happy|'ll clean|'ll process)|i['\u2019]?m happy)\b/i;
  const META_PHRASES = [
    'provide the transcript',
    'provide the text',
    'provide the voice',
    'i will process',
    'i understand',
    "i'll clean",
    'i will clean',
    'as an ai',
    'happy to help',
  ];
  function looksLikeMetaResponse(raw, out) {
    if (!out) return false;
    const s = out.toLowerCase();
    if (META_START.test(out)) return true;
    for (let i = 0; i < META_PHRASES.length; i++) {
      if (s.indexOf(META_PHRASES[i]) !== -1) return true;
    }
    // Word-overlap heuristic — if the output is much longer than the input and
    // shares very few words with it, the model likely responded TO the prompt
    // rather than transforming it. Only fires when the input has >= 5 words so
    // short fragments don't produce false positives.
    const inWords = new Set((raw.toLowerCase().match(/[a-z]+/g) || []).filter(w => w.length > 2));
    const outWords = (out.toLowerCase().match(/[a-z]+/g) || []);
    if (inWords.size >= 5 && outWords.length > inWords.size * 3) {
      let overlap = 0;
      for (const w of outWords) if (inWords.has(w)) overlap++;
      if (overlap < Math.max(2, Math.floor(inWords.size / 3))) return true;
    }
    return false;
  }

  // ── Blur levels ──────────────────────────────────────────────────────────
  const FADE_WORDS = 6;
  const LEVELS = [
    { blur: 0,   opacity: 1.0  },
    { blur: 0.5, opacity: 0.9  },
    { blur: 1.5, opacity: 0.78 },
    { blur: 2.5, opacity: 0.65 },
    { blur: 3.5, opacity: 0.52 },
    { blur: 4.5, opacity: 0.42 },
  ];

  const DOT = '<span class="vtt-dot"></span>';

  function buildLiveHtml(text, showDot) {
    const dot = showDot !== false ? DOT : '';
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (!words.length) return dot;

    if (words.length <= FADE_WORDS) {
      return words.map((w, i) => {
        const pos = words.length - 1 - i;
        const lvl = LEVELS[Math.min(pos, LEVELS.length - 1)];
        return `<span style="filter:blur(${lvl.blur}px);opacity:${lvl.opacity}">${esc(w)}</span>`;
      }).join(' ') + dot;
    }
    const blurred = words.slice(0, -FADE_WORDS).map(esc).join(' ');
    const fading = words.slice(-FADE_WORDS).map((w, i) => {
      const pos = FADE_WORDS - 1 - i;
      const lvl = LEVELS[Math.min(pos, LEVELS.length - 1)];
      return `<span style="filter:blur(${lvl.blur}px);opacity:${lvl.opacity}">${esc(w)}</span>`;
    }).join(' ');
    return `<span class="vtt-blurred">${blurred}</span> ${fading}${dot}`;
  }

  // Render combined cleaned (unblurred, slightly bolder) + live (blur fade + dot)
  // inside the interim div, so the user sees one flowing transcript where the
  // cleaned prefix grows in place as chunks come back from /clean.
  function renderInterim(el, cleanedText, liveText, showDot) {
    const cleanedPart = cleanedText && cleanedText.trim()
      ? `<span class="vtt-cleaned">${esc(cleanedText.trim())}</span>`
      : '';
    const liveHtml = buildLiveHtml(liveText || '', showDot);
    const joiner   = cleanedPart && liveHtml ? ' ' : '';
    el.innerHTML   = cleanedPart + joiner + liveHtml;
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Main init ────────────────────────────────────────────────────────────
  function init(opts) {
    // Accept either a DOM node or a string id for any element option.
    // This lets reply.cc pass freshly-created textarea nodes that have no id.
    function resolveEl(val) {
      if (!val) return null;
      return (val instanceof HTMLElement) ? val : document.getElementById(val);
    }

    const targetEl  = resolveEl(opts.target);
    if (!targetEl) { console.error('VoiceToText: target textarea not found:', opts.target); return; }

    // cleanUrl: null or '' means skip /clean entirely and append raw text.
    const cleanUrl = (opts.cleanUrl === null || opts.cleanUrl === '') ? null : (opts.cleanUrl || '/clean');
    const lang     = opts.lang || 'en-AU';

    // Resolve or auto-create elements
    let btnEl     = resolveEl(opts.button);
    let interimEl = resolveEl(opts.interim);
    let statusEl  = resolveEl(opts.status);
    let labelEl   = resolveEl(opts.label);

    if (!interimEl) {
      interimEl = document.createElement('div');
      interimEl.className = 'vtt-interim';
      targetEl.parentNode.insertBefore(interimEl, targetEl);
    } else {
      interimEl.classList.add('vtt-interim');
    }

    if (!btnEl) {
      btnEl = document.createElement('button');
      btnEl.type = 'button';
      btnEl.className = 'vtt-btn';
      btnEl.setAttribute('aria-label', 'Record voice');
      btnEl.innerHTML = MIC_SVG;
      targetEl.parentNode.insertBefore(btnEl, interimEl);
    } else {
      btnEl.classList.add('vtt-btn');
      btnEl.innerHTML = MIC_SVG;
    }

    // Edit hint — appears after cleanup to nudge users to review
    const hintEl = document.createElement('div');
    hintEl.className = 'vtt-edit-hint';
    hintEl.textContent = 'Tap the text above to review and edit';
    targetEl.parentNode.insertBefore(hintEl, targetEl.nextSibling);

    // ── State ────────────────────────────────────────────────────────────
    let recognition        = null;
    let isRecording        = false;
    let shippedResultCount = 0;    // how many event.results have been sent to /clean
    let pendingFinal       = '';   // finalised speech since the last ship, waiting to go
    let pendingFinalCount  = 0;    // how many event.results contribute to pendingFinal
    let cleanedSoFar       = '';
    let preVoice           = '';
    let wakeLock           = null;
    let tickTimer          = null;
    let lastSpeechAt       = 0;    // ms timestamp of last onresult (any result)
    let lastShipAt         = 0;    // ms timestamp of last shipped chunk
    let latestInterim      = '';   // current in-progress interim speech, for async re-renders
    // Every chunk shipped to /clean is appended here; the cleaned response
    // removes its own entry. Tracked as a list (not a single string) so two
    // overlapping round-trips can never produce a visual gap.
    let inFlightChunks     = [];   // array of { id, text } — drives the blur display
    let inFlightSeq        = 0;    // monotonic id, doubles as chunk sequence number
    // Seq-based ordered commit: chunks must be appended to the textarea in the
    // order they were spoken, even if /clean responses arrive out of order.
    let committedSeq       = 0;    // seq of the last chunk flushed to appendCleanedChunk
    let pendingCommit      = new Map(); // seq → cleanedText, waiting for in-order flush
    function inFlightJoined() {
      return inFlightChunks.map(c => c.text).join(' ');
    }

    // Walk pendingCommit and flush any chunks that are now in order.
    function tryFlushCommits() {
      while (pendingCommit.has(committedSeq + 1)) {
        committedSeq++;
        const text = pendingCommit.get(committedSeq);
        pendingCommit.delete(committedSeq);
        appendCleanedChunk(text);
      }
    }

    function startTickTimer() {
      lastShipAt = Date.now();
      tickTimer = setInterval(() => {
        if (!isRecording) return;
        const text = pendingFinal.trim();
        if (!text) return;
        const now = Date.now();
        const paused = (now - lastSpeechAt) >= PAUSE_MS;
        const maxed  = (now - lastShipAt)   >= MAX_CHUNK_MS;
        // Paused with a substantial chunk → ship.
        // Paused but chunk still tiny → wait (don't send "um" alone).
        // Maxed → ship regardless (safety fallback for continuous talkers).
        if (maxed || (paused && text.length >= MIN_CHUNK_CHARS)) shipChunk();
      }, TICK_MS);
    }

    function stopTickTimer() {
      if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    }

    function shipChunk() {
      const text = pendingFinal.trim();
      if (!text) return;
      shippedResultCount += pendingFinalCount;
      pendingFinal       = '';
      pendingFinalCount  = 0;
      lastShipAt         = Date.now();
      cleanChunkInBackground(text);
    }

    function setCaption(voiceText) {
      targetEl.value = preVoice
        ? preVoice.trimEnd() + '\n\n' + voiceText
        : voiceText;
    }

    function initRecognition() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {
        if (statusEl) statusEl.textContent = 'Speech recognition not supported — use Chrome or Safari.';
        btnEl.disabled = true;
        return false;
      }

      // Tear down any existing recognition object before creating a fresh one.
      // Reusing the same object across multiple start/stop cycles (especially
      // after an audio-session interruption from the file picker) leaves the
      // object in a broken state where onend fires at unexpected times and
      // recognition.start() silently fails. Always start fresh.
      if (recognition) {
        recognition.onresult = null;
        recognition.onerror  = null;
        recognition.onend    = null;
        try { recognition.abort(); } catch {}
        recognition = null;
      }

      recognition = new SR();
      recognition.continuous     = true;
      recognition.interimResults = true;
      recognition.lang           = lang;

      recognition.onresult = (event) => {
        lastSpeechAt      = Date.now();
        pendingFinal      = '';
        pendingFinalCount = 0;
        let interim       = '';
        // Only iterate results that haven't been shipped to /clean yet.
        // event.results is cumulative across the life of this recognition
        // object — starting at shippedResultCount gives us the true delta.
        for (let i = shippedResultCount; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            pendingFinal     += event.results[i][0].transcript + ' ';
            pendingFinalCount = i - shippedResultCount + 1;
          } else {
            interim += event.results[i][0].transcript;
          }
        }
        latestInterim = interim;
        // Combined display: cleaned prefix (unblurred) + live blur tail.
        // preVoice is whatever the textarea held before recording started —
        // it's already "clean" so it sits in the cleaned prefix too.
        // inFlightChunks are chunks shipped to /clean but not yet returned —
        // keep them visible (blurred) so the user never sees words vanish
        // during a round-trip. Two overlapping round-trips are safe because
        // each chunk has its own entry; only its own response removes it.
        const cleanedPrefix = [preVoice.trim(), cleanedSoFar.trim()]
          .filter(Boolean).join(' ');
        const liveText = [inFlightJoined(), pendingFinal + interim]
          .filter(s => s && s.trim()).join(' ');
        renderInterim(interimEl, cleanedPrefix, liveText);
      };

      recognition.onerror = (e) => {
        if (e.error === 'no-speech') return;
        const msg = e.error.toLowerCase().includes('call') || e.error === 'not-allowed'
          ? 'Mic busy — close other apps using audio and try again'
          : 'Error: ' + e.error;
        if (statusEl) statusEl.textContent = msg;
        stopVoice(false);
      };

      recognition.onend = () => {
        if (isRecording) {
          // Ship any un-shipped finalised speech before we tear down this
          // recognition object — initRecognition() resets event.results to
          // empty, so we'd lose the context otherwise.
          const chunkText = pendingFinal.trim();
          pendingFinal      = '';
          pendingFinalCount = 0;
          // Fresh recognition object's event.results starts at 0, so our
          // cursor must too. Also clear stale interim so it doesn't bleed
          // into the next session's live display.
          shippedResultCount = 0;
          latestInterim      = '';
          // Create a fresh recognition object before restarting — calling
          // recognition.start() on the same object that just fired onend
          // throws InvalidStateError on Chrome/Android, which the catch
          // would previously swallow by calling stopVoice(false), silently
          // killing recording on the first pause.
          if (!initRecognition()) { stopVoice(false); return; }
          try {
            recognition.start();
          } catch {
            // Recognition is in a bad state (e.g. tab backgrounded).
            // Clean up gracefully rather than leaving isRecording true with dead recognition.
            stopVoice(false);
            return;
          }
          if (chunkText) {
            lastShipAt = Date.now();
            cleanChunkInBackground(chunkText);
          }
        }
      };
      return true;
    }

    function appendCleanedChunk(text) {
      cleanedSoFar = cleanedSoFar ? cleanedSoFar.trimEnd() + '\n\n' + text : text;
      setCaption(cleanedSoFar);
      if (isRecording) {
        // While recording the textarea stays hidden — the interim div is
        // the single display. Re-render so the just-cleaned chunk appears
        // as unblurred prefix in place of the blurred pending speech.
        const cleanedPrefix = [preVoice.trim(), cleanedSoFar.trim()]
          .filter(Boolean).join(' ');
        renderInterim(interimEl, cleanedPrefix, pendingFinal + latestInterim, true);
      } else {
        // Post-stop finalisation — textarea is the edit surface now.
        if (statusEl) statusEl.textContent = '';
        targetEl.style.display = '';
        targetEl.classList.remove('vtt-chunk-new');
        void targetEl.offsetWidth;
        targetEl.classList.add('vtt-chunk-new');
        setTimeout(() => targetEl.classList.remove('vtt-chunk-new'), 450);
      }
    }

    function cleanChunkInBackground(text) {
      // Noise guard — skip chunks that have no transcribable content.
      // Prevents the LLM from seeing a near-empty prompt and responding
      // with meta-chatter like "please provide the transcript".
      if (!text || !/[a-zA-Z]/.test(text)) return;

      // noCleanup mode: cleanUrl is null/'' — bypass all /clean fetches.
      if (!cleanUrl) {
        const id = ++inFlightSeq;
        inFlightChunks.push({ id, text });
        pendingCommit.set(id, text);
        inFlightChunks = inFlightChunks.filter(c => c.id !== id);
        tryFlushCommits();
        return;
      }

      const id = ++inFlightSeq;
      inFlightChunks.push({ id, text });
      const remove = () => {
        inFlightChunks = inFlightChunks.filter(c => c.id !== id);
      };

      // Per-chunk 15 s timeout: if the /clean response never arrives, force-
      // commit the raw text with a marker and unblock any later chunks waiting
      // in pendingCommit. The `settled` flag prevents both the timeout and the
      // real response from committing the same chunk.
      let settled = false;
      const timeoutId = setTimeout(() => {
        if (settled) return;
        settled = true;
        pendingCommit.set(id, '\u2026 ' + text);
        // Advance committedSeq through any gap so tryFlushCommits can proceed.
        if (committedSeq < id - 1) committedSeq = id - 1;
        tryFlushCommits();
      }, 15000);

      fetch(cleanUrl, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text }),
      })
        .then(res => {
          // Treat 4xx/5xx as errors so the catch block handles status UX.
          // (fetch only rejects on network errors, not HTTP error codes.)
          if (!res.ok) throw new Error('http_' + res.status);
          return res.json();
        })
        .then(data => {
          if (settled) return; // timeout already committed this chunk
          clearTimeout(timeoutId);
          settled = true;
          const cleaned = data && data.cleaned;
          // Client-side meta-response guard — defence-in-depth for when
          // an older backend container is still running the prior prompt.
          // If the cleaned output looks like the model answered the prompt
          // instead of transforming the text, fall back to the raw input.
          const result = (cleaned && !looksLikeMetaResponse(text, cleaned)) ? cleaned : text;
          pendingCommit.set(id, result);
          tryFlushCommits();
        })
        .catch(err => {
          if (settled) return; // timeout already committed this chunk
          clearTimeout(timeoutId);
          settled = true;
          // Surface a transient status message so the user knows something
          // was paused (413 payload too large, 429 rate limit, network drop).
          if (statusEl) {
            statusEl.textContent = 'Cleanup paused \u2014 raw text shown.';
            setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 3000);
          }
          // Graceful fallback: use raw text if the clean request fails.
          pendingCommit.set(id, text);
          tryFlushCommits();
        })
        .finally(() => {
          // remove() ALWAYS runs — regardless of happy path, error, or throw
          // inside the .then body — so inFlightChunks never leaks an entry.
          remove();
        });
    }

    async function startVoice() {
      // Disable the button immediately to prevent a double-tap race condition.
      // If the user taps twice quickly, the second tap arrives while this
      // function is suspended at the wakeLock await. Without this guard the
      // second tap calls stopVoice(), then the first tap resumes and calls
      // recognition.start() on a stopped object — zombie state.
      btnEl.disabled = true;

      // Always recreate the recognition object on each new session so stale
      // state from a previous session (or an audio-session interruption from
      // the file picker) can't corrupt this one.
      if (!initRecognition()) { btnEl.disabled = false; return; }
      preVoice           = targetEl.value;
      shippedResultCount = 0;
      pendingFinal       = '';
      pendingFinalCount  = 0;
      cleanedSoFar       = '';
      inFlightChunks     = [];
      inFlightSeq        = 0;
      committedSeq       = 0;
      pendingCommit      = new Map();
      isRecording        = true;
      lastSpeechAt       = Date.now();
      if ('wakeLock' in navigator) {
        try { wakeLock = await navigator.wakeLock.request('screen'); } catch {}
      }

      // Guard: if stopVoice() was called while we were awaiting wakeLock,
      // isRecording will have been set to false. Abort the start sequence.
      if (!isRecording) { btnEl.disabled = false; return; }

      btnEl.classList.add('vtt-recording');
      btnEl.innerHTML = STOP_SVG;
      if (labelEl)  labelEl.textContent  = 'Tap to stop';
      if (statusEl) statusEl.textContent = 'Listening\u2026';
      hintEl.classList.remove('vtt-visible');
      targetEl.classList.remove('vtt-textarea-highlight');
      targetEl.style.display  = 'none';
      interimEl.style.display = 'block';
      // Show existing text as the cleaned prefix so it flows seamlessly
      // into the live speech that's about to come.
      latestInterim = '';
      renderInterim(interimEl, preVoice, '', true);
      recognition.start();
      startTickTimer();
      btnEl.disabled = false;
    }

    async function stopVoice(clean) {
      if (clean === undefined) clean = true;
      stopTickTimer();
      isRecording = false;
      if (wakeLock) { wakeLock.release(); wakeLock = null; }
      if (recognition) {
        // Detach handlers before stopping so a delayed onend from this stop()
        // call cannot trigger an unwanted restart or corrupt a new session
        // that starts immediately after (e.g. rapid tap, or file picker close).
        recognition.onresult = null;
        recognition.onerror  = null;
        recognition.onend    = null;
        try { recognition.stop(); } catch {}
      }
      btnEl.classList.remove('vtt-recording');
      btnEl.innerHTML = MIC_SVG;
      if (labelEl)  labelEl.textContent  = 'Talk to text';

      const remainingRaw = pendingFinal.trim();
      // "Nothing heard" only if there's genuinely no captured speech — including
      // chunks already shipped to /clean but not yet returned (inFlightChunks)
      // or returned but waiting for in-order commit (pendingCommit).
      const hasInFlight = inFlightChunks.length > 0 || pendingCommit.size > 0;
      if (!clean || (!remainingRaw && !cleanedSoFar && !hasInFlight)) {
        interimEl.style.display = 'none';
        targetEl.style.display  = '';
        if (clean && statusEl) statusEl.textContent = 'Nothing heard \u2014 try again.';
        return;
      }

      interimEl.style.display = 'none';
      targetEl.style.display  = '';
      btnEl.disabled = true;

      // Wrap all post-stop cleanup in try/finally so the button is ALWAYS
      // re-enabled when stopVoice() completes, even if the fetch throws or
      // the response cannot be parsed.
      try {
        if (remainingRaw && /[a-zA-Z]/.test(remainingRaw)) {
          // There's a final chunk not yet cleaned — send it synchronously.
          if (statusEl) statusEl.textContent = 'Tidying up\u2026';
          if (cleanUrl) {
            try {
              const res  = await fetch(cleanUrl, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ text: remainingRaw }),
              });
              const data = res.ok ? await res.json() : null;
              const cleaned = data && data.cleaned;
              if (cleaned && !looksLikeMetaResponse(remainingRaw, cleaned)) {
                appendCleanedChunk(cleaned);
              } else {
                appendCleanedChunk(remainingRaw);
              }
            } catch {
              appendCleanedChunk(remainingRaw);
            }
          } else {
            // noCleanup mode — append raw text directly.
            appendCleanedChunk(remainingRaw);
          }
          if (statusEl) statusEl.textContent = '';
        } else if (hasInFlight) {
          if (statusEl) statusEl.textContent = 'Tidying up\u2026';
        } else {
          if (statusEl) statusEl.textContent = '';
        }
      } finally {
        // Button ALWAYS re-enabled — regardless of network state or thrown errors.
        btnEl.disabled = false;
      }

      // Highlight textarea and show edit hint
      if (targetEl.value.trim()) {
        targetEl.classList.add('vtt-textarea-highlight');
        hintEl.classList.add('vtt-visible');
        setTimeout(() => {
          targetEl.classList.remove('vtt-textarea-highlight');
          hintEl.classList.remove('vtt-visible');
        }, 5000);
        // Dismiss hint on focus — { once: true } auto-removes the listener so
        // the explicit removeEventListener call is not needed.
        targetEl.addEventListener('focus', function dismissHint() {
          targetEl.classList.remove('vtt-textarea-highlight');
          hintEl.classList.remove('vtt-visible');
        }, { once: true });
      }
    }

    btnEl.addEventListener('click', () => {
      isRecording ? stopVoice() : startVoice();
    });

    // Return control object for programmatic use
    return { start: startVoice, stop: stopVoice, isRecording: () => isRecording };
  }

  // ── Export ──────────────────────────────────────────────────────────────
  window.VoiceToText = { init };
})();
