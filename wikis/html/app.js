
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('wiki-theme'); } catch (e) {}
  var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (saved === 'dark' || (saved === null && prefersDark)) root.classList.add('dark');

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-btn');
    if (btn) btn.addEventListener('click', function () {
      root.classList.toggle('dark');
      try { localStorage.setItem('wiki-theme', root.classList.contains('dark') ? 'dark' : 'light'); } catch (e) {}
    });
    var menu = document.getElementById('menu-btn');
    if (menu) menu.addEventListener('click', function () { document.body.classList.toggle('nav-open'); });
    document.addEventListener('click', function (e) {
      if (document.body.classList.contains('nav-open') &&
          !e.target.closest('.sidebar') && e.target.id !== 'menu-btn' &&
          !e.target.closest('#menu-btn')) {
        document.body.classList.remove('nav-open');
      }
    });

    // Build the on-page TOC from h2/h3 in the article.
    var article = document.querySelector('.article');
    var toc = document.querySelector('.toc');
    if (!article || !toc) return;
    var heads = article.querySelectorAll('h2[id], h3[id]');
    if (heads.length < 2) { toc.remove(); return; }
    var html = '<div class="toc-title">On this page</div>';
    heads.forEach(function (h) {
      var lvl = h.tagName === 'H3' ? ' lvl-3' : '';
      var text = h.textContent.replace('\u00b6', '').trim();
      html += '<a class="toc-link' + lvl + '" href="#' + h.id + '">' + text + '</a>';
    });
    toc.innerHTML = html;

    var links = toc.querySelectorAll('a');
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          if (byId[en.target.id]) byId[en.target.id].classList.add('active');
        }
      });
    }, { rootMargin: '-72px 0px -70% 0px', threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  });
})();
