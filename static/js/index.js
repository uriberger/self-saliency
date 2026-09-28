// Theme toggle + motion-respecting teaser playback.

(function () {
  'use strict';

  /* ---------------- theme ---------------- */

  var root = document.documentElement;
  var btn = document.getElementById('theme-toggle');
  var KEY = 'selfsal-theme';

  var stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) { /* private mode */ }
  if (stored === 'light' || stored === 'dark') root.setAttribute('data-theme', stored);

  function current() {
    var set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  if (btn) {
    btn.addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
      btn.setAttribute('aria-pressed', String(next === 'dark'));
    });
    btn.setAttribute('aria-pressed', String(current() === 'dark'));
  }

  /* ---------------- teaser ----------------
     The teaser loops silently, which is motion nobody asked for. Start it only
     when it is on screen and the visitor has not asked for reduced motion; the
     controls stay available either way. */

  var video = document.getElementById('teaser');
  if (!video) return;

  var calm = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (calm.matches) return;

  if (!('IntersectionObserver' in window)) {
    video.autoplay = true;
    video.play().catch(function () { /* blocked by the browser; controls remain */ });
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        video.play().catch(function () { /* blocked; controls remain */ });
      } else if (!video.paused) {
        video.pause();
      }
    });
  }, { threshold: 0.25 });

  io.observe(video);
})();
