/* Text-Classification-Lab — interactions de la page */
(function () {
  "use strict";

  var L = window.LAB;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var css = function (n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); };
  var pct1 = function (v) { return (v * 100).toFixed(1).replace(".", ",") + " %"; };
  var fr = function (v) { return String(v).replace(".", ","); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]; }); };
  var reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var narrow = window.matchMedia("(max-width: 560px)");
  var nm = function (it) { return narrow.matches && it.short ? it.short : it.name; };

  function hexA(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map(function (c) { return c + c; }).join("");
    var n = parseInt(h, 16);
    return "rgba(" + (n >> 16) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  var ICON_OK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  var ICON_KO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  /* ============================================================== Thème */
  var restylers = [];
  (function () {
    var btn = $("#themeToggle");
    function sync() {
      var dark = document.documentElement.getAttribute("data-theme") === "dark";
      btn.setAttribute("aria-label", dark ? "Passer en thème clair" : "Passer en thème sombre");
    }
    sync();
    btn.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      sync();
      restylers.forEach(function (f) { f(); });
    });
  })();

  /* ====================================================== Démo en direct */
  (function () {
    var input = $("#demoInput"), chips = $("#demoChips"),
        box = $("#demoVerdict"), why = $("#demoWhy"), status = $("#demoStatus");
    var ready = false, timer = null;

    L.demo.forEach(function (d) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = d.tag;
      b.addEventListener("click", function () { input.value = d.text; input.focus(); render(); });
      chips.appendChild(b);
    });

    function empty() {
      box.innerHTML = '<div class="verdict-empty">Le verdict s\'affiche ici dès que tu écris quelque chose.</div>';
      why.innerHTML = "";
    }

    function fail(msg) {
      status.textContent = "modèle non chargé";
      box.innerHTML = '<div class="verdict-error">' + msg + "</div>";
      why.innerHTML = "";
      input.disabled = true;
      chips.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
    }

    function render() {
      if (!ready) return;
      var text = input.value.trim();
      if (!text) { empty(); return; }

      var r = window.SpamModel.predict(text);
      var isSpam = r.label === "spam";
      var p = r.probability;

      box.innerHTML =
        '<div class="verdict-box ' + (isSpam ? "is-spam" : "is-ham") + '">' +
          '<div class="verdict-top">' +
            '<span class="verdict-label">' + (isSpam ? "Spam" : "Message normal") + "</span>" +
            '<span class="verdict-pct">' + pct1(p) + "</span>" +
          "</div>" +
          '<div class="gauge"><span style="width:' + (p * 100).toFixed(1) + '%"></span></div>' +
          '<div class="verdict-scale"><span>message normal</span><span>probabilité de spam</span><span>spam</span></div>' +
        "</div>";

      var top = r.parts.slice(0, 6);
      if (!top.length) {
        why.innerHTML = '<p class="why-none">Aucun de ces mots ne figure dans le vocabulaire de 5 000 termes appris à l\'entraînement. Le modèle répond alors par défaut.</p>';
        return;
      }
      var max = Math.abs(top[0].weight) || 1;
      why.innerHTML =
        "<h3>Les mots qui ont pesé</h3>" +
        top.map(function (t) {
          var w = Math.min(50, Math.abs(t.weight) / max * 50);
          var pos = t.weight > 0;
          return '<div class="why-row">' +
            '<span class="why-word">' + esc(t.term) + "</span>" +
            '<span class="why-track"><span class="why-bar ' + (pos ? "pos" : "neg") + '" style="width:' + w.toFixed(1) + '%"></span></span>' +
            '<span class="why-val">' + (pos ? "+" : "−") + Math.abs(t.weight).toFixed(2).replace(".", ",") + "</span>" +
          "</div>";
        }).join("") +
        '<div class="why-axis"><span>← vers « message normal »</span><span>vers « spam » →</span></div>';
    }

    empty();
    input.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(render, 120);
    });

    window.SpamModel.load("model/spam_model.json").then(function (m) {
      ready = true;
      status.textContent = "régression logistique · " + m.terms.length.toLocaleString("fr-FR") + " mots";
      render();
    }).catch(function () {
      if (location.protocol === "file:") {
        fail("Ouverte depuis le disque, la page ne peut pas charger le modèle : le navigateur bloque la lecture de fichiers locaux. Ouvre la version en ligne, ou lance <code>python -m http.server</code> dans le dossier <code>docs</code>.");
      } else {
        fail("Le fichier du modèle n'a pas pu être chargé. Vérifie que <code>docs/model/spam_model.json</code> est bien présent.");
      }
    });
  })();

  /* ========================================================= Graphiques */
  if (window.Chart) {
    Chart.defaults.font.family = css("--f-body") || "system-ui";
    Chart.defaults.font.size = 13;
    if (reduceMotion) Chart.defaults.animation = false;
  }

  var valueLabels = {
    id: "valueLabels",
    afterDatasetsDraw: function (chart) {
      var ctx = chart.ctx, meta = chart.getDatasetMeta(0), data = chart.data.datasets[0].data;
      ctx.save();
      ctx.font = "700 13px " + Chart.defaults.font.family;
      ctx.fillStyle = css("--ink");
      ctx.textBaseline = "middle";
      meta.data.forEach(function (bar, i) { ctx.fillText(pct1(data[i]), bar.x + 8, bar.y); });
      ctx.restore();
    }
  };

  var pointLabels = {
    id: "pointLabels",
    afterDatasetsDraw: function (chart) {
      var ctx = chart.ctx, meta = chart.getDatasetMeta(0), pts = chart.data.datasets[0].data;
      ctx.save();
      ctx.font = "600 13px " + Chart.defaults.font.family;
      ctx.fillStyle = css("--ink");
      meta.data.forEach(function (el, i) {
        var p = pts[i], gap = 16;
        if (p.pos === "left")  { ctx.textAlign = "right";  ctx.textBaseline = "middle"; ctx.fillText(p.label, el.x - gap, el.y); }
        if (p.pos === "right") { ctx.textAlign = "left";   ctx.textBaseline = "middle"; ctx.fillText(p.label, el.x + gap, el.y); }
        if (p.pos === "below") { ctx.textAlign = "center"; ctx.textBaseline = "top";    ctx.fillText(p.label, el.x, el.y + gap - 2); }
      });
      ctx.restore();
    }
  };

  function tooltipStyle() {
    return { backgroundColor: css("--ink"), titleColor: css("--paper"), bodyColor: css("--paper"),
             padding: 10, cornerRadius: 8, displayColors: false, titleFont: { weight: "700" } };
  }

  /* -------------------------------------------- Bloc graphique + détail */
  function Lab(o) {
    var metric = "f";
    var selected = o.items.reduce(function (b, i) { return i.f > b.f ? i : b; }, o.items[0]).id;
    var chart = null, scatter = null;

    var seg = $(o.metricsEl);
    Object.keys(L.metrics).forEach(function (k) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.metric = k; b.textContent = L.metrics[k].short;
      b.setAttribute("aria-pressed", k === metric ? "true" : "false");
      seg.appendChild(b);
    });
    seg.addEventListener("click", function (e) {
      var b = e.target.closest("button"); if (!b) return;
      metric = b.dataset.metric;
      seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
      refresh(true);
    });

    function sorted() {
      var a = o.items.slice();
      if (o.sort) a.sort(function (x, y) { return y[metric] - x[metric]; });
      return a;
    }
    function barColor(it) {
      var base = css(L.metrics[metric].cssVar);
      return it.id === selected ? base : hexA(base, 0.45);
    }

    function buildBars() {
      var arr = sorted();
      chart = new Chart($(o.canvas), {
        type: "bar",
        data: { labels: arr.map(nm), datasets: [{
          data: arr.map(function (i) { return i[metric]; }),
          backgroundColor: arr.map(barColor),
          borderRadius: { topRight: 5, bottomRight: 5 }, borderSkipped: false, barThickness: 24 }] },
        options: {
          indexAxis: "y", maintainAspectRatio: false,
          layout: { padding: { right: narrow.matches ? 54 : 66 } },
          animation: reduceMotion ? false : { duration: 550, easing: "easeOutQuart" },
          onClick: function (_, els) { if (els.length) select(sorted()[els[0].index].id); },
          onHover: function (e, els) { e.native.target.style.cursor = els.length ? "pointer" : "default"; },
          scales: {
            x: { min: 0, max: 1, grid: { color: css("--line-soft") }, border: { display: false },
                 ticks: { color: css("--ink-3"), stepSize: 0.25, callback: function (v) { return Math.round(v * 100) + " %"; } } },
            y: { grid: { display: false }, border: { color: css("--line") },
                 ticks: { color: css("--ink"), font: { size: 14, weight: "600" } } }
          },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tooltipStyle(), { callbacks: {
              title: function (c) { return c[0].label; },
              label: function (c) { return L.metrics[metric].label + " : " + pct1(c.parsed.x); } } })
          }
        },
        plugins: [valueLabels]
      });
    }

    function buildScatter() {
      var pos = { nb: "below", rf: "right", lr: "left", svm: "left" };
      var pts = o.items.map(function (i) { return { x: i.r, y: i.p, label: nm(i), id: i.id, pos: pos[i.id] || "right" }; });
      scatter = new Chart($(o.scatterCanvas), {
        type: "scatter",
        data: { datasets: [{ data: pts,
          pointRadius: pts.map(function (p) { return p.id === selected ? 11 : 8; }),
          pointBackgroundColor: pts.map(function (p) { return p.id === selected ? css("--accent") : hexA(css("--accent"), 0.42); }),
          pointBorderColor: css("--surface"), pointBorderWidth: 2, pointHoverRadius: 12 }] },
        options: {
          maintainAspectRatio: false,
          animation: reduceMotion ? false : { duration: 350 },
          onClick: function (_, els) { if (els.length) select(pts[els[0].index].id); },
          onHover: function (e, els) { e.native.target.style.cursor = els.length ? "pointer" : "default"; },
          scales: {
            x: { min: 0.75, max: 0.95, title: { display: true, text: "Recall : part des spams trouvés", color: css("--ink-2") },
                 grid: { color: css("--line-soft") }, border: { display: false },
                 ticks: { color: css("--ink-3"), stepSize: 0.05, callback: function (v) { return Math.round(v * 100) + " %"; } } },
            y: { min: 0.88, max: 1.02, title: { display: true, text: "Precision : fausses alertes évitées", color: css("--ink-2") },
                 grid: { color: css("--line-soft") }, border: { display: false },
                 ticks: { color: css("--ink-3"), stepSize: 0.04, callback: function (v) { return v > 1.001 ? "" : Math.round(v * 100) + " %"; } } }
          },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tooltipStyle(), { callbacks: {
              title: function (c) { return c[0].raw.label; },
              label: function (c) { return "Precision " + pct1(c.raw.y) + " · Recall " + pct1(c.raw.x); } } })
          }
        },
        plugins: [pointLabels]
      });
    }

    var picker = $(o.pickerEl), detail = $(o.detailEl);
    o.items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.id = it.id;
      b.innerHTML = "<span>" + esc(it.name) + "</span><small></small>";
      picker.appendChild(b);
    });
    picker.addEventListener("click", function (e) {
      var b = e.target.closest("button"); if (b) select(b.dataset.id);
    });
    function select(id) { selected = id; refresh(false); }

    function renderDetail() {
      var it = o.items.filter(function (x) { return x.id === selected; })[0];
      detail.innerHTML =
        "<b>" + esc(it.name) + "</b><p>" + esc(o.noteOf(it)) + "</p>" +
        (o.extra ? o.extra(it) : "") +
        '<div class="mini">' + ["acc", "p", "r", "f"].map(function (k) {
          return "<div><strong>" + pct1(it[k]) + "</strong><span>" + L.metrics[k].short + "</span></div>";
        }).join("") + "</div>";
    }

    function refresh(animate) {
      $(o.helpEl).textContent = L.metrics[metric].label + " : " + L.metrics[metric].help;
      picker.querySelectorAll("button").forEach(function (b) {
        var it = o.items.filter(function (x) { return x.id === b.dataset.id; })[0];
        b.setAttribute("aria-pressed", b.dataset.id === selected ? "true" : "false");
        b.querySelector("small").textContent = L.metrics[metric].short + " " + pct1(it[metric]);
      });
      renderDetail();
      if (chart) {
        var arr = sorted();
        chart.data.labels = arr.map(nm);
        chart.data.datasets[0].data = arr.map(function (i) { return i[metric]; });
        chart.data.datasets[0].backgroundColor = arr.map(barColor);
        chart.update(animate ? undefined : "none");
      }
      if (scatter) {
        var ds = scatter.data.datasets[0], pts = ds.data;
        ds.pointRadius = pts.map(function (p) { return p.id === selected ? 11 : 8; });
        ds.pointBackgroundColor = pts.map(function (p) { return p.id === selected ? css("--accent") : hexA(css("--accent"), 0.42); });
        scatter.update("none");
      }
    }

    function restyle() {
      [chart, scatter].forEach(function (c) { if (c) c.destroy(); });
      chart = scatter = null;
      buildBars(); if (o.scatterCanvas) buildScatter();
      refresh(false);
    }

    buildBars();
    if (o.scatterCanvas) buildScatter();
    refresh(false);
    if (narrow.addEventListener) narrow.addEventListener("change", restyle);
    restylers.push(restyle);

    return { metric: function () { return metric; } };
  }

  /* ------------------------------------------------- Tableaux triables */
  function SortableTable(sel, caption, cols, rows) {
    var table = $(sel), key = null, dir = 1, best = {};
    cols.forEach(function (c) {
      if (c.best) best[c.key] = Math.max.apply(null, rows.map(function (r) { return r[c.key]; }));
    });
    function render() {
      var data = rows.slice();
      if (key) data.sort(function (a, b) {
        var x = a[key], y = b[key];
        return (typeof x === "string" ? x.localeCompare(y) : (x === y ? 0 : x < y ? -1 : 1)) * dir;
      });
      table.innerHTML = "<caption>" + caption + "</caption><thead><tr>" +
        cols.map(function (c) {
          var s = key === c.key ? (dir === 1 ? "ascending" : "descending") : "none";
          return '<th scope="col" aria-sort="' + s + '"><button type="button" data-key="' + c.key + '">' + c.label + "</button></th>";
        }).join("") + "</tr></thead><tbody>" +
        data.map(function (r) {
          return "<tr>" + cols.map(function (c) {
            var v = c.fmt ? c.fmt(r[c.key], r) : r[c.key];
            if (c.key === "name") return '<th scope="row">' + esc(String(v)) + "</th>";
            var cls = c.best && r[c.key] === best[c.key] ? "best" : (c.cls || "");
            return "<td" + (cls ? ' class="' + cls + '"' : "") + ">" + v + "</td>";
          }).join("") + "</tr>";
        }).join("") + "</tbody>";
    }
    table.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-key]"); if (!b) return;
      if (key === b.dataset.key) dir = -dir; else { key = b.dataset.key; dir = key === "name" ? 1 : -1; }
      render();
      var again = table.querySelector('button[data-key="' + key + '"]');
      if (again) again.focus();
    });
    render();
  }

  /* ============================================================== Spam */
  function confusion(it) {
    var c = it.cm;
    return '<div class="cm"><h4>Sur les ' + L.test.total.toLocaleString("fr-FR") + " SMS de test (" +
      L.test.ham + " normaux, " + L.test.spam + " spams)</h4><div class=\"cm-grid\">" +
      '<div class="cm-cell good"><b>' + c[0] + "</b><span>messages normaux laissés passer</span></div>" +
      '<div class="cm-cell ' + (c[1] ? "err" : "good") + '"><b>' + c[1] + "</b><span>messages normaux bloqués à tort</span></div>" +
      '<div class="cm-cell ' + (c[2] ? "err" : "good") + '"><b>' + c[2] + "</b><span>spams non détectés</span></div>" +
      '<div class="cm-cell good"><b>' + c[3] + "</b><span>spams bloqués</span></div>" +
      "</div></div>";
  }

  var spamLab = Lab({
    items: L.spam, sort: true,
    canvas: "#spamBars", scatterCanvas: "#spamScatter",
    metricsEl: "#spamMetrics", helpEl: "#spamHelp", pickerEl: "#spamPicker", detailEl: "#spamDetail",
    noteOf: function (it) { return it.note; },
    extra: confusion
  });

  (function () {
    var seg = $("#spamView"), bars = $("#spamBarsBox"), sc = $("#spamScatterBox"), metrics = $("#spamMetrics");
    seg.addEventListener("click", function (e) {
      var b = e.target.closest("button"); if (!b) return;
      var on = b.dataset.view === "scatter";
      seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
      bars.hidden = on; sc.hidden = !on;
      metrics.style.visibility = on ? "hidden" : "visible";
      $("#spamHelp").textContent = on
        ? "Plus un point est haut, moins le modèle bloque de vrais messages. Plus il est à droite, moins il laisse passer de spams. Clique sur un point pour le détail."
        : L.metrics[spamLab.metric()].label + " : " + L.metrics[spamLab.metric()].help;
      window.dispatchEvent(new Event("resize"));
    });
  })();

  SortableTable("#spamTable", "Tous les scores, classe « spam »", [
    { key: "name", label: "Modèle" },
    { key: "acc", label: "Accuracy",  best: true, fmt: pct1 },
    { key: "p",   label: "Precision", best: true, fmt: pct1 },
    { key: "r",   label: "Recall",    best: true, fmt: pct1 },
    { key: "f",   label: "F1-score",  best: true, fmt: pct1 },
    { key: "fp",  label: "Faux spams",   cls: "cfg", fmt: function (v) { return v; } },
    { key: "fn",  label: "Spams ratés",  cls: "cfg", fmt: function (v) { return v; } }
  ], L.spam.map(function (r) { return Object.assign({}, r, { fp: r.cm[1], fn: r.cm[2] }); }));

  /* ============================================================== IMDb */
  function cfgLine(it) {
    var c = it.cfg;
    return '<p class="cfg">Batch ' + c.batch + " · " + c.epochs + " epochs · learning rate " + fr(c.lr) +
      " · dropout " + (c.dropout ? "oui" : "non") + " · loss pondérée " + (c.weighted ? "oui" : "non") + "</p>";
  }

  Lab({
    items: L.imdb, sort: false,
    canvas: "#imdbBars",
    metricsEl: "#imdbMetrics", helpEl: "#imdbHelp", pickerEl: "#imdbPicker", detailEl: "#imdbDetail",
    noteOf: function (it) { return it.change; },
    extra: cfgLine
  });

  (function () {
    $("#imdbLadder").innerHTML = L.imdb.map(function (it, i) {
      var d = i === 0 ? null : (it.f - L.imdb[i - 1].f) * 100;
      var cls = d === null ? "flat" : d > 0.05 ? "up" : d < -0.05 ? "down" : "flat";
      var txt = d === null ? "F1 " + pct1(it.f) : (d > 0 ? "+" : "−") + Math.abs(d).toFixed(1).replace(".", ",") + " pt";
      var sub = d === null ? "référence" : "de F1 sur l'étape d'avant";
      return "<li><b>" + esc(it.name) + '</b><p class="what">' + esc(it.change) + '</p><div class="delta ' + cls + '">' + txt + "<small>" + sub + "</small></div></li>";
    }).join("");
  })();

  SortableTable("#imdbTable", "Toutes les expériences", [
    { key: "name", label: "Expérience" },
    { key: "batch",    label: "Batch",         cls: "cfg", fmt: function (v) { return v; } },
    { key: "epochs",   label: "Epochs",        cls: "cfg", fmt: function (v) { return v; } },
    { key: "lr",       label: "Learning rate", cls: "cfg", fmt: fr },
    { key: "dropout",  label: "Dropout",       cls: "cfg", fmt: function (v) { return v ? "oui" : "non"; } },
    { key: "weighted", label: "Loss pondérée", cls: "cfg", fmt: function (v) { return v ? "oui" : "non"; } },
    { key: "acc", label: "Accuracy",  best: true, fmt: pct1 },
    { key: "p",   label: "Precision", best: true, fmt: pct1 },
    { key: "r",   label: "Recall",    best: true, fmt: pct1 },
    { key: "f",   label: "F1-score",  best: true, fmt: pct1 }
  ], L.imdb.map(function (r) {
    return Object.assign({}, r, { batch: r.cfg.batch, epochs: r.cfg.epochs, lr: r.cfg.lr,
                                  dropout: r.cfg.dropout ? 1 : 0, weighted: r.cfg.weighted ? 1 : 0 });
  }));

  /* --------------------------------------------------- Schéma du réseau */
  (function () {
    var W = 900, H = 272, xs = [90, 270, 450, 630, 810], ys = [46, 86, 126, 166, 206];
    var names = [["Entrée", "10 000 poids TF-IDF"], ["Couche 1", "256 neurones · BatchNorm"],
                 ["Couche 2", "128 neurones"], ["Couche 3", "64 neurones"], ["Sortie", "1 nombre"]];
    var counts = [5, 5, 5, 5, 1], NS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Schéma du réseau : 10 000 entrées, puis des couches de 256, 128 et 64 neurones, puis une sortie unique.");
    function el(tag, a, txt) {
      var e = document.createElementNS(NS, tag);
      Object.keys(a).forEach(function (k) { e.setAttribute(k, a[k]); });
      if (txt) e.textContent = txt;
      return e;
    }
    var layerY = function (l) { return counts[l] === 1 ? [126] : ys; };
    for (var l = 0; l < 4; l++) {
      layerY(l).forEach(function (y1) {
        layerY(l + 1).forEach(function (y2) {
          svg.appendChild(el("line", { x1: xs[l], y1: y1, x2: xs[l + 1], y2: y2, "class": "link" }));
        });
      });
    }
    for (var k = 0; k < 5; k++) {
      (function (k) {
        layerY(k).forEach(function (y) {
          svg.appendChild(el("circle", { cx: xs[k], cy: y, r: 10, "class": "node" + (k === 4 ? " out" : "") }));
        });
        svg.appendChild(el("text", { x: xs[k], y: 248, "text-anchor": "middle", "class": "big" }, names[k][0]));
        svg.appendChild(el("text", { x: xs[k], y: 267, "text-anchor": "middle" }, names[k][1]));
      })(k);
    }
    $("#netSvg").appendChild(svg);
  })();

  /* =============================================== Toi contre le modèle */
  (function () {
    var ex = L.examples, i = 0, you = 0, bot = 0, answered = false;
    var msg = $("#gameMsg"), count = $("#gameCount"), row = $("#guessRow"),
        rev = $("#reveal"), foot = $("#gameFoot"), sub = $(".game-sub");
    var label = function (l) { return l === "spam" ? "Spam" : "Message normal"; };
    var pill = function (l) { return '<span class="pill ' + l + '">' + label(l) + "</span>"; };
    var mark = function (ok) { return '<span class="mark ' + (ok ? "ok" : "ko") + '">' + (ok ? ICON_OK + "Bonne réponse" : ICON_KO + "Erreur") + "</span>"; };

    function show() {
      var e = ex[i];
      answered = false;
      count.textContent = "Message " + (i + 1) + " sur " + ex.length;
      sub.textContent = "Ce SMS est-il un spam ?";
      msg.hidden = false; row.hidden = false; foot.hidden = false;
      rev.hidden = true; rev.innerHTML = "";
      msg.textContent = e.text;
      msg.style.borderLeftColor = "var(--ink-3)";
      row.querySelectorAll("button").forEach(function (b) { b.disabled = false; });
    }

    function guess(g) {
      if (answered) return;
      answered = true;
      var e = ex[i], youOk = g === e.truth, botOk = e.model === e.truth, last = i === ex.length - 1;
      if (youOk) you++;
      if (botOk) bot++;
      row.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
      msg.style.borderLeftColor = e.truth === "spam" ? "var(--bad)" : "var(--ok)";
      rev.innerHTML =
        '<div class="rv-row"><span>Vraie réponse</span>' + pill(e.truth) + "</div>" +
        '<div class="rv-row"><span>Ton choix</span><span class="rv-who">' + pill(g) + mark(youOk) + "</span></div>" +
        '<div class="rv-row"><span>Le SVM a répondu</span><span class="rv-who">' + pill(e.model) + mark(botOk) + "</span></div>" +
        '<div class="next-row"><button class="btn btn-primary" type="button" id="gameNext">' + (last ? "Voir le score" : "Message suivant") + "</button></div>";
      rev.hidden = false;
      var next = $("#gameNext");
      next.focus();
      next.addEventListener("click", function () { if (last) end(); else { i++; show(); } });
    }

    function end() {
      count.textContent = "Terminé";
      sub.textContent = "Sur ces cinq messages :";
      msg.hidden = true; row.hidden = true; foot.hidden = true;
      rev.hidden = false;
      rev.innerHTML =
        '<div class="score"><div><b>' + you + "/5</b><span>pour toi</span></div>" +
        "<div><b>" + bot + "/5</b><span>pour le SVM</span></div></div>" +
        '<p class="game-foot" style="border:0;padding-top:2px">Ces cinq messages ont été choisis pour montrer deux erreurs : le score du SVM ici ne reflète pas sa vraie qualité. Sur les 1 115 SMS de test, il en classe bien 1\u00a0094.</p>' +
        '<div class="next-row"><button class="btn btn-primary" type="button" id="gameAgain">Rejouer</button></div>';
      var again = $("#gameAgain");
      again.addEventListener("click", function () { i = 0; you = 0; bot = 0; show(); });
      again.focus();
    }

    row.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-guess]");
      if (b) guess(b.dataset.guess);
    });
    show();
  })();
})();
