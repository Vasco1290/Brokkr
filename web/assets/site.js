/* Brokkr website v0 (and the design prototypes). Optional enhancements only: every page works without
   this file (web/site_checks.py checks that nothing is hidden until it runs).
   1. The theme picker (choice kept in this browser only; no tracking; default follows the system).
   2. The bench's condition keys: move the needle, update the readout (and, in the prototypes, show one
      condition's table).
   3. Sorting the compare page's table (without it, the table stays in name order).
   The script never computes a number: every text and angle it shows was written into the page from the labels. */
(function () {
  "use strict";
  var root = document.documentElement;
  var KEY = "brokkr-theme";

  var picker = document.getElementById("theme-picker");
  if (picker) {
    picker.value = root.getAttribute("data-theme") || "auto";
    picker.addEventListener("change", function () {
      var v = picker.value;
      if (v === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", v);
      try { if (v === "auto") localStorage.removeItem(KEY); else localStorage.setItem(KEY, v); } catch (e) { /* no storage: fine */ }
    });
  }

  // 3. Sortable tables (the compare page): a button in each heading sorts the rows by their cells'
  //    data-sort values (numbers as numbers, words alphabetically; empty values, e.g. a failed build, last).
  function sortKey(cell) { var v = cell.getAttribute("data-sort"); return v === null ? cell.textContent : v; }
  function sortBy(table, col, th) {
    var dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
    var heads = table.tHead.rows[0].cells;
    for (var h = 0; h < heads.length; h++) heads[h].removeAttribute("aria-sort");
    th.setAttribute("aria-sort", dir);
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = sortKey(a.cells[col]), y = sortKey(b.cells[col]);
      if (x === "" || y === "") return (x === "") - (y === "");
      var nx = parseFloat(x), ny = parseFloat(y);
      var c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
      return dir === "ascending" ? c : -c;
    });
    for (var r = 0; r < rows.length; r++) body.appendChild(rows[r]);
  }
  var sortables = document.querySelectorAll("table.sortable");
  for (var s = 0; s < sortables.length; s++) {
    (function (table) {
      var heads = table.tHead.rows[0].cells;
      for (var c = 0; c < heads.length; c++) {
        (function (th, col) {
          var b = document.createElement("button");
          b.type = "button";
          while (th.firstChild) b.appendChild(th.firstChild);
          th.appendChild(b);
          b.addEventListener("click", function () { sortBy(table, col, th); });
        })(heads[c], c);
      }
    })(sortables[s]);
  }

  var radios = document.querySelectorAll('input[name="cond"]');
  var needle = document.getElementById("meter-needle");
  if (!radios.length || !needle) return;
  var band = document.getElementById("meter-band");
  var tables = document.querySelectorAll(".cond-table");

  function text(id, value) { var el = document.getElementById(id); if (el) el.textContent = value; }

  function show(input) {
    var d = input.dataset;
    needle.style.transform = "rotate(" + (-parseFloat(d.angle)) + "deg)";
    band.setAttribute("d", d.band);
    text("r-cost", d.costText);
    text("r-interval", d.intervalText);
    text("r-cond", d.label);
    var v = document.getElementById("r-verdict");
    v.className = "verdict v-" + d.verdictKind;
    v.textContent = "";
    var icon = document.createElement("span");
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = d.icon;
    v.appendChild(icon);
    v.appendChild(document.createTextNode(" " + d.verdict));
    for (var i = 0; i < tables.length; i++) tables[i].hidden = tables[i].getAttribute("data-cond") !== input.value;
  }

  for (var i = 0; i < radios.length; i++) radios[i].addEventListener("change", function () { show(this); });

  // On load the needle rests at zero, then swings to the reading and settles (it jumps if motion is reduced).
  var start = document.querySelector('input[name="cond"]:checked') || radios[0];
  needle.style.transition = "none";
  needle.style.transform = "rotate(" + (-parseFloat(needle.getAttribute("data-rest"))) + "deg)";
  void needle.getBoundingClientRect();
  needle.style.transition = "";
  requestAnimationFrame(function () { show(start); });
})();
