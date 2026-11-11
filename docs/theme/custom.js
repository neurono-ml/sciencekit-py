// sciencekit for Python — theme interactions (sidebar brand, Light/Dark picker, font scale, scroll reveal)

(function () {
  function cleanThemeList() {
    var list = document.getElementById('mdbook-theme-list');
    if (!list || list.dataset.skCleaned) return;
    list.dataset.skCleaned = '1';
    ['default_theme', 'rust', 'navy', 'ayu'].forEach(function (suffix) {
      var button = document.getElementById('mdbook-theme-' + suffix);
      if (button && button.parentElement) button.parentElement.remove();
    });
    var coal = document.getElementById('mdbook-theme-coal');
    if (coal) coal.textContent = 'Dark';
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', cleanThemeList);
  } else {
    cleanThemeList();
  }
})();

(function () {
  function injectBrand() {
    var scrollbox = document.querySelector('.sidebar-scrollbox');
    if (scrollbox && !scrollbox.querySelector('.brand')) {
      var brand = document.createElement('div');
      brand.className = 'brand';
      brand.innerHTML =
        '<span class="brand-mark">py</span>' +
        '<span><span class="brand-name">sciencekit-py</span>' +
        '<span class="brand-tagline">Python bindings</span></span>';
      scrollbox.insertBefore(brand, scrollbox.firstChild);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectBrand);
  } else {
    injectBrand();
  }
})();

document.addEventListener('DOMContentLoaded', function () {
  var reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );
    reveals.forEach(function (el) { observer.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add('is-visible'); });
  }
});

(function () {
  var STORAGE_KEY = 'sk-py-font-scale';
  var MINIMUM_SCALE = 0;
  var MAXIMUM_SCALE = 2;
  var DEFAULT_SCALE = 0;

  function readScale() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      var parsed = parseInt(raw, 10);
      if (parsed >= MINIMUM_SCALE && parsed <= MAXIMUM_SCALE) return parsed;
    } catch (error) { /* ignore storage errors */ }
    return DEFAULT_SCALE;
  }

  function applyScale(scale) {
    document.documentElement.setAttribute('data-sk-font-scale', String(scale));
    try { window.localStorage.setItem(STORAGE_KEY, String(scale)); }
    catch (error) { /* ignore storage errors */ }
    document.querySelectorAll('.sk-font-btn').forEach(function (button) {
      var action = button.getAttribute('data-sk-font-action');
      if (action === 'reset') {
        button.setAttribute('aria-pressed', scale === DEFAULT_SCALE ? 'true' : 'false');
      } else {
        button.disabled =
          (action === 'decrease' && scale <= MINIMUM_SCALE) ||
          (action === 'increase' && scale >= MAXIMUM_SCALE);
      }
    });
  }

  function injectControls() {
    var bar = document.querySelector('#mdbook-menu-bar .left-buttons, .menu-bar .left-buttons');
    if (!bar || bar.querySelector('.sk-font-btn')) return;
    var group = document.createElement('div');
    group.className = 'sk-font-group';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'Text size');
    group.innerHTML =
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="decrease" title="Decrease text size" aria-label="Decrease text size">A−</button>' +
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="reset" title="Reset text size" aria-label="Reset text size">A</button>' +
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="increase" title="Increase text size" aria-label="Increase text size">A+</button>';
    bar.appendChild(group);
    group.addEventListener('click', function (event) {
      var button = event.target.closest('.sk-font-btn');
      if (!button) return;
      var action = button.getAttribute('data-sk-font-action');
      var scale = readScale();
      if (action === 'decrease') scale = Math.max(MINIMUM_SCALE, scale - 1);
      else if (action === 'increase') scale = Math.min(MAXIMUM_SCALE, scale + 1);
      else scale = DEFAULT_SCALE;
      applyScale(scale);
    });
    applyScale(readScale());
  }

  applyScale(readScale());
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectControls);
  } else {
    injectControls();
  }
})();
