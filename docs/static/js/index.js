// ==== Gaussians on Fire — project page scripts ====
// Placeholder for interactive behavior (carousels, video controls, etc.).

document.addEventListener('DOMContentLoaded', function () {
  // Bulma navbar burger toggle (kept here for when a navbar is added).
  const burgers = Array.prototype.slice.call(
    document.querySelectorAll('.navbar-burger'), 0
  );
  burgers.forEach(function (el) {
    el.addEventListener('click', function () {
      const target = document.getElementById(el.dataset.target);
      el.classList.toggle('is-active');
      if (target) target.classList.toggle('is-active');
    });
  });

  // ==== Baseline comparison: scene / modality toggles + synced playback ====
  const grid = document.getElementById('comparison-grid');
  const controls = document.getElementById('comparison-controls');
  if (grid && controls) {
    const VIDEO_DIR = './static/videos/';
    const oursVideo = grid.querySelector('[data-method="ours"]');
    const baseVideo = document.getElementById('baseline-video');
    const baseCaption = document.getElementById('baseline-caption');
    const select = document.getElementById('baseline-select');
    const state = { scene: '3_2', baseline: select ? select.value : '4dgs_cvpr' };

    function load(video, method) {
      const want = VIDEO_DIR + method + '_' + state.scene + '_rgb.mp4';
      if (video.getAttribute('src') !== want) {
        video.setAttribute('src', want);
        video.load();
      }
      video.currentTime = 0;
      const p = video.play();
      if (p && p.catch) p.catch(function () {}); // ignore autoplay rejections
    }

    function update() {
      load(oursVideo, 'ours');
      load(baseVideo, state.baseline);
    }

    // Scene toggle
    controls.addEventListener('click', function (e) {
      const btn = e.target.closest('button');
      if (!btn || !btn.dataset.scene || state.scene === btn.dataset.scene) return;
      state.scene = btn.dataset.scene;
      btn.closest('.buttons').querySelectorAll('button').forEach(function (b) {
        b.classList.remove('is-dark', 'is-selected');
      });
      btn.classList.add('is-dark', 'is-selected');
      update();
    });

    // Baseline selector
    if (select) {
      select.addEventListener('change', function () {
        state.baseline = select.value;
        if (baseCaption) baseCaption.textContent = select.options[select.selectedIndex].text;
        update();
      });
    }

    update(); // initial load (scene 1, first baseline)
  }

  // ==== Dataset gallery: thumbnail wall + synchronized triptych lightbox ====
  const dgrid = document.getElementById('dataset-grid');
  const lightbox = document.getElementById('dataset-lightbox');
  if (dgrid && lightbox) {
    const GAL = './static/videos/gallery/';
    const SCENES = [
      { n: 1, cams: [GAL + 'capture_03/scene_0001/cam_0.mp4', GAL + 'capture_03/scene_0001/cam_1.mp4', GAL + 'capture_03/scene_0001/cam_2.mp4'] },
      { n: 2, cams: [GAL + 'capture_03/scene_0002/cam_0.mp4', GAL + 'capture_03/scene_0002/cam_1.mp4', GAL + 'capture_03/scene_0002/cam_2.mp4'] },
      { n: 3, cams: [GAL + 'capture_03/scene_0003/cam_0.mp4', GAL + 'capture_03/scene_0003/cam_1.mp4', GAL + 'capture_03/scene_0003/cam_2.mp4'] },
      { n: 4, cams: [GAL + 'capture_03/scene_0004/cam_0.mp4', GAL + 'capture_03/scene_0004/cam_1.mp4', GAL + 'capture_03/scene_0004/cam_2.mp4'] },
      { n: 5, cams: [GAL + 'capture_03/scene_0005/cam_0.mp4', GAL + 'capture_03/scene_0005/cam_1.mp4', GAL + 'capture_03/scene_0005/cam_2.mp4'] },
      { n: 6, cams: [GAL + 'capture_03/scene_0006/cam_0.mp4', GAL + 'capture_03/scene_0006/cam_1.mp4', GAL + 'capture_03/scene_0006/cam_2.mp4'] },
      { n: 7, cams: [GAL + 'capture_04/scene_0001/cam_0.mp4', GAL + 'capture_04/scene_0001/cam_1.mp4', GAL + 'capture_04/scene_0001/cam_2.mp4'] },
      { n: 8, cams: [GAL + 'capture_04/scene_0002/cam_0.mp4', GAL + 'capture_04/scene_0002/cam_1.mp4', GAL + 'capture_04/scene_0002/cam_2.mp4'] },
      { n: 9, cams: [GAL + 'capture_04/scene_0003/cam_0.mp4', GAL + 'capture_04/scene_0003/cam_1.mp4', GAL + 'capture_04/scene_0003/cam_2.mp4'] },
      { n: 10, cams: [GAL + 'capture_04/scene_0005/cam_0.mp4', GAL + 'capture_04/scene_0005/cam_1.mp4', GAL + 'capture_04/scene_0005/cam_2.mp4'] },
      { n: 11, cams: [GAL + 'capture_04/scene_0006/cam_0.mp4', GAL + 'capture_04/scene_0006/cam_1.mp4', GAL + 'capture_04/scene_0006/cam_2.mp4'] },
      { n: 12, cams: [GAL + 'capture_04/scene_0007/cam_0.mp4', GAL + 'capture_04/scene_0007/cam_1.mp4', GAL + 'capture_04/scene_0007/cam_2.mp4'] },
      { n: 13, cams: [GAL + 'capture_04/scene_0008/cam_0.mp4', GAL + 'capture_04/scene_0008/cam_1.mp4', GAL + 'capture_04/scene_0008/cam_2.mp4'] },
      { n: 14, cams: [GAL + 'capture_04/scene_0009/cam_0.mp4', GAL + 'capture_04/scene_0009/cam_1.mp4', GAL + 'capture_04/scene_0009/cam_2.mp4'] },
      { n: 15, cams: [GAL + 'capture_04/scene_0010/cam_0.mp4', GAL + 'capture_04/scene_0010/cam_1.mp4', GAL + 'capture_04/scene_0010/cam_2.mp4'] },
      { n: 16, cams: [GAL + 'capture_05/scene_0003/cam_0.mp4', GAL + 'capture_05/scene_0003/cam_1.mp4', GAL + 'capture_05/scene_0003/cam_2.mp4'] },
      { n: 17, cams: [GAL + 'capture_05/scene_0006/cam_0.mp4', GAL + 'capture_05/scene_0006/cam_1.mp4', GAL + 'capture_05/scene_0006/cam_2.mp4'] },
    ];

    // Poster frame for scene n, camera k (extracted offline at ~2s).
    function posterPath(n, k) {
      return GAL + 'posters/g' + String(n).padStart(2, '0') + '_cam' + k + '.jpg';
    }

    const lbVideos = [0, 1, 2].map(function (i) { return document.getElementById('lb-cam' + i); });
    const lbCaption = document.getElementById('dataset-lb-caption');
    let current = 0;
    let driftTimer = null;
    const visible = new Set();   // thumbnails currently in the viewport

    function isOpen() { return lightbox.getAttribute('aria-hidden') === 'false'; }

    // --- Build the thumbnail wall (one camera per scene) ---
    SCENES.forEach(function (sc, i) {
      const btn = document.createElement('button');
      btn.className = 'dataset-thumb';
      btn.type = 'button';
      btn.setAttribute('aria-label', 'Open scene ' + sc.n);

      const v = document.createElement('video');
      v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none';
      v.poster = posterPath(sc.n, 1);
      v.dataset.src = sc.cams[1];   // middle camera as the wall preview

      const label = document.createElement('span');
      label.className = 'dataset-thumb-label';
      label.textContent = sc.n;

      btn.appendChild(v);
      btn.appendChild(label);
      btn.addEventListener('click', function () { openLightbox(i); });
      dgrid.appendChild(btn);
    });

    const thumbVideos = Array.prototype.slice.call(dgrid.querySelectorAll('video'));

    // --- Lazy-load + only play thumbnails that are on screen ---
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        const v = e.target;
        if (e.isIntersecting) {
          visible.add(v);
          if (!v.getAttribute('src') && v.dataset.src) v.setAttribute('src', v.dataset.src);
          if (!isOpen()) safePlay(v);
        } else {
          visible.delete(v);
          v.pause();
        }
      });
    }, { rootMargin: '150px' });
    thumbVideos.forEach(function (v) { io.observe(v); });

    function safePlay(v) { const p = v.play(); if (p && p.catch) p.catch(function () {}); }

    // --- Lightbox ---
    let loadToken = 0;   // invalidates pending loads when the user navigates on

    // Resolves once the video has buffered enough to play (or errored/stalled,
    // so one bad file can't wedge the lightbox).
    function whenReady(v) {
      return new Promise(function (resolve) {
        if (v.readyState >= 3) { resolve(); return; }
        let timer = null;
        function done() {
          clearTimeout(timer);
          v.removeEventListener('canplay', done);
          v.removeEventListener('error', done);
          resolve();
        }
        v.addEventListener('canplay', done);
        v.addEventListener('error', done);
        timer = setTimeout(done, 8000);
      });
    }

    // Resolves once playback has actually begun rendering frames, so the
    // placeholders never uncover a stalled or still-black view.
    function whenPlaying(v) {
      return new Promise(function (resolve) {
        let timer = null;
        function done() {
          clearTimeout(timer);
          v.removeEventListener('playing', done);
          resolve();
        }
        v.addEventListener('playing', done);
        timer = setTimeout(done, 1000);
      });
    }

    function loadScene() {
      const sc = SCENES[current];
      const token = ++loadToken;
      lightbox.classList.add('is-loading');
      lbVideos.forEach(function (v, k) {
        // No poster here: the video paints its own first frame once decoded,
        // so the dimmed still matches frame 0 exactly and playback starts
        // without a content jump.
        v.preload = 'auto';
        v.src = sc.cams[k];
        v.load();
      });
      if (lbCaption) lbCaption.textContent = 'Scene ' + sc.n + ' / ' + SCENES.length;

      // Hold playback until all three views are buffered, start them together
      // behind the placeholders, and only then reveal — so the crossfade
      // uncovers videos that are already in motion.
      Promise.all(lbVideos.map(whenReady)).then(function () {
        if (token !== loadToken || !isOpen()) return;
        playAll();
        return Promise.all(lbVideos.map(whenPlaying)).then(function () {
          if (token !== loadToken || !isOpen()) return;
          lightbox.classList.remove('is-loading');
        });
      });
    }

    function playAll() {
      lbVideos.forEach(function (v) {
        try { v.currentTime = 0; } catch (err) {}
        safePlay(v);
      });
    }

    // Keep the three views locked: snap followers to the master if they drift.
    function startDriftSync() {
      stopDriftSync();
      driftTimer = setInterval(function () {
        const m = lbVideos[0];
        if (m.readyState < 2) return;
        for (let k = 1; k < 3; k++) {
          const v = lbVideos[k];
          if (v.readyState < 2 || v.seeking) continue;
          if (Math.abs(v.currentTime - m.currentTime) > 0.1) v.currentTime = m.currentTime;
        }
      }, 400);
    }
    function stopDriftSync() { if (driftTimer) { clearInterval(driftTimer); driftTimer = null; } }

    function openLightbox(i) {
      current = i;
      thumbVideos.forEach(function (v) { v.pause(); });  // free decoders for the triptych
      loadScene();
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
      startDriftSync();
    }

    function closeLightbox() {
      loadToken++;   // cancel any in-flight scene load
      lightbox.setAttribute('aria-hidden', 'true');
      lightbox.classList.remove('is-loading');
      document.body.style.overflow = '';
      stopDriftSync();
      lbVideos.forEach(function (v) { v.pause(); v.removeAttribute('src'); v.load(); });
      visible.forEach(safePlay);   // resume the thumbnails still on screen
    }

    function go(delta) {
      current = (current + delta + SCENES.length) % SCENES.length;
      loadScene();
    }

    document.getElementById('dataset-close').addEventListener('click', closeLightbox);
    document.getElementById('dataset-prev').addEventListener('click', function () { go(-1); });
    document.getElementById('dataset-next').addEventListener('click', function () { go(1); });
    lightbox.addEventListener('click', function (e) { if (e.target === lightbox) closeLightbox(); });
    document.addEventListener('keydown', function (e) {
      if (!isOpen()) return;
      if (e.key === 'Escape') closeLightbox();
      else if (e.key === 'ArrowLeft') go(-1);
      else if (e.key === 'ArrowRight') go(1);
    });
  }
});
