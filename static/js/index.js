// Motion-respecting teaser playback.
//
// The clip loops silently, which is motion nobody asked for. Start it only when
// it is on screen and the visitor has not asked for reduced motion; the native
// controls stay available either way.

(function () {
  'use strict';

  var video = document.getElementById('teaser');
  if (!video) return;

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  function play() {
    video.play().catch(function () { /* blocked by the browser; controls remain */ });
  }

  if (!('IntersectionObserver' in window)) {
    video.autoplay = true;
    play();
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        play();
      } else if (!video.paused) {
        video.pause();
      }
    });
  }, { threshold: 0.25 });

  io.observe(video);
})();
