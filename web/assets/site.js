/* Brokkr website v0, design prototype. Optional enhancements only: every page works without this file.
   1. The theme picker (choice kept in this browser only; no tracking; default follows the system).
   2. The bench's condition switches: move the needle, update the readout, show one condition's table.
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
