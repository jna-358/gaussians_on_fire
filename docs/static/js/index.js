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
});
