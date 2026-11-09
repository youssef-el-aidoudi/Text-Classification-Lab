/* Classificateur de spam qui tourne dans le navigateur.
   Il refait, en JavaScript, exactement le calcul de scikit-learn :
   TF-IDF (vocabulaire figé à l'entraînement) puis régression logistique.
   Les poids viennent de model/spam_model.json, produit par export_model_for_web.py. */
window.SpamModel = (function () {
  "use strict";

  var M = null, index = null;

  /* Équivalent du token_pattern \b\w\w+\b de scikit-learn :
     les suites d'au moins deux caractères de mot, en minuscules. */
  var TOKEN = /[\p{L}\p{N}_]{2,}/gu;

  function tokenize(text) {
    return String(text).toLowerCase().match(TOKEN) || [];
  }

  function load(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error("modèle introuvable");
      return r.json();
    }).then(function (json) {
      M = json;
      index = new Map();
      for (var i = 0; i < M.terms.length; i++) index.set(M.terms[i], i);
      return M;
    });
  }

  function predict(text) {
    if (!M) throw new Error("modèle non chargé");

    // 1. Compter les mots connus du vocabulaire
    var tokens = tokenize(text);
    var counts = new Map(), known = 0;
    tokens.forEach(function (t) {
      var j = index.get(t);
      if (j === undefined) return;
      known++;
      counts.set(j, (counts.get(j) || 0) + 1);
    });

    // 2. TF x IDF, puis normalisation L2
    var raw = [], norm = 0;
    counts.forEach(function (c, j) {
      var v = c * M.idf[j];
      raw.push([j, v]);
      norm += v * v;
    });
    norm = Math.sqrt(norm) || 1;

    // 3. Produit scalaire avec les coefficients du modèle
    var logit = M.intercept, parts = [];
    raw.forEach(function (pair) {
      var j = pair[0], x = pair[1] / norm, w = M.coef[j] * x;
      logit += w;
      parts.push({ term: M.terms[j], weight: w, count: counts.get(j) });
    });
    parts.sort(function (a, b) { return Math.abs(b.weight) - Math.abs(a.weight); });

    return {
      label: logit > 0 ? "spam" : "ham",
      probability: 1 / (1 + Math.exp(-logit)),
      logit: logit,
      parts: parts,
      nTokens: tokens.length,
      nKnown: known
    };
  }

  return { load: load, predict: predict, tokenize: tokenize,
           info: function () { return M; } };
})();
