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
      @keyframes vttPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.15); }
        50%       { box-shadow: 0 0 0 5px rgba(220,38,38,0.08); }
      }
    `;
    document.head.appendChild(style);
  }

  // ── SVGs ─────────────────────────────────────────────────────────────────
  const MIC_SVG  = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>';
  const STOP_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

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

  function setInterimText(el, text) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (!words.length) { el.innerHTML = ''; return; }

    if (words.length <= FADE_WORDS) {
      el.innerHTML = words.map((w, i) => {
        const pos = words.length - 1 - i;
        const lvl = LEVELS[Math.min(pos, LEVELS.length - 1)];
        return `<span style="filter:blur(${lvl.blur}px);opacity:${lvl.opacity}">${esc(w)}</span>`;
      }).join(' ');
    } else {
      const blurred = words.slice(0, -FADE_WORDS).map(esc).join(' ');
      const fading = words.slice(-FADE_WORDS).map((w, i) => {
        const pos = FADE_WORDS - 1 - i;
        const lvl = LEVELS[Math.min(pos, LEVELS.length - 1)];
        return `<span style="filter:blur(${lvl.blur}px);opacity:${lvl.opacity}">${esc(w)}</span>`;
      }).join(' ');
      el.innerHTML = `<span class="vtt-blurred">${blurred}</span> ${fading}`;
    }
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Main init ────────────────────────────────────────────────────────────
  function init(opts) {
    const targetEl  = document.getElementById(opts.target);
    if (!targetEl) { console.error('VoiceToText: target textarea not found:', opts.target); return; }

    const cleanUrl = opts.cleanUrl || '/clean';
    const lang     = opts.lang || 'en-AU';

    // Resolve or auto-create elements
    let btnEl     = opts.button  ? document.getElementById(opts.button)  : null;
    let interimEl = opts.interim ? document.getElementById(opts.interim) : null;
    let statusEl  = opts.status  ? document.getElementById(opts.status)  : null;
    let labelEl   = opts.label   ? document.getElementById(opts.label)   : null;

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
    }

    // ── State ────────────────────────────────────────────────────────────
    let recognition  = null;
    let isRecording  = false;
    let accumulated  = '';
    let sessionFinal = '';
    let preVoice     = '';
    let wakeLock     = null;

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
      recognition = new SR();
      recognition.continuous     = true;
      recognition.interimResults = true;
      recognition.lang           = lang;

      recognition.onresult = (event) => {
        sessionFinal = '';
        let interim  = '';
        for (let i = 0; i < event.results.length; i++) {
          if (event.results[i].isFinal) sessionFinal += event.results[i][0].transcript + ' ';
          else interim += event.results[i][0].transcript;
        }
        const fullText = accumulated + sessionFinal + interim;
        setCaption(fullText);
        setInterimText(interimEl, fullText);
      };

      recognition.onerror = (e) => {
        if (e.error === 'no-speech') return;
        if (statusEl) statusEl.textContent = 'Error: ' + e.error;
        stopVoice(false);
      };

      recognition.onend = () => {
        if (isRecording) {
          accumulated  += sessionFinal;
          sessionFinal  = '';
          recognition.start();
        }
      };
      return true;
    }

    async function startVoice() {
      if (!recognition && !initRecognition()) return;
      preVoice     = targetEl.value;
      accumulated  = '';
      sessionFinal = '';
      isRecording  = true;
      if ('wakeLock' in navigator) {
        try { wakeLock = await navigator.wakeLock.request('screen'); } catch {}
      }
      btnEl.classList.add('vtt-recording');
      btnEl.innerHTML = STOP_SVG;
      if (labelEl)  labelEl.textContent  = 'Tap to stop';
      if (statusEl) statusEl.textContent = 'Listening\u2026';
      targetEl.style.display  = 'none';
      interimEl.style.display = 'block';
      interimEl.innerHTML     = '';
      recognition.start();
    }

    async function stopVoice(clean) {
      if (clean === undefined) clean = true;
      isRecording = false;
      if (wakeLock) { wakeLock.release(); wakeLock = null; }
      if (recognition) recognition.stop();
      btnEl.classList.remove('vtt-recording');
      btnEl.innerHTML = MIC_SVG;
      if (labelEl)  labelEl.textContent  = 'Talk to text';
      interimEl.style.display = 'none';
      targetEl.style.display  = '';

      const raw = (accumulated + sessionFinal).trim();
      if (!clean || !raw) {
        if (clean && statusEl) statusEl.textContent = 'Nothing heard \u2014 try again.';
        return;
      }

      if (statusEl) statusEl.textContent = 'Tidying up\u2026';
      btnEl.disabled = true;
      try {
        const res  = await fetch(cleanUrl, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ text: raw }),
        });
        const data = await res.json();
        setCaption(data.cleaned || raw);
        if (statusEl) statusEl.textContent = '';
      } catch {
        if (statusEl) statusEl.textContent = '';
      } finally {
        btnEl.disabled = false;
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
