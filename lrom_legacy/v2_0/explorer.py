"""Export a trained v2 cross-section LROM as an offline HTML explorer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from . import (
    MASS_PION,
    __version__,
    _cross_section_cache,
    _packed_effective_interaction_features,
    _smatrix_from_packed_coordinates,
    _solve_packed_coordinates,
)


HTML_SCHEMA_VERSION = 1
_PARAMETER_NAMES = (
    "Vv",
    "Wv",
    "Wd",
    "Vso",
    "Rv",
    "Rd",
    "Rso",
    "av",
    "ad",
    "aso",
)
_PARAMETER_LABELS = {
    "Vv": "Vv",
    "Wv": "Wv",
    "Wd": "Wd",
    "Vso": "Vso",
    "Rv": "Rv",
    "Rd": "Rd",
    "Rso": "Rso",
    "av": "av",
    "ad": "ad",
    "aso": "aso",
}
_PARAMETER_GROUPS = {
    "Vv": "Volume",
    "Wv": "Volume",
    "Rv": "Volume",
    "av": "Volume",
    "Wd": "Surface",
    "Rd": "Surface",
    "ad": "Surface",
    "Vso": "Spin-Orbit",
    "Rso": "Spin-Orbit",
    "aso": "Spin-Orbit",
}
_REQUIRED_TRAINING_OPTIONS = {
    "basis_size": 6,
    "predictor": "effective-interaction",
    "predictor_count": 6,
    "observable": "cross_section",
}
_DEFAULT_OPTIONS = {
    "title": "Cross Section Explorer with L-ROM",
    "subtitle": None,
    "footer_note": None,
    "allow_extrapolation": False,
    "slider_ranges": {},
    "slider_steps": {},
    "slider_formats": {},
    "parameter_labels": {},
    "parameter_groups": {},
    "parameter_order": list(_PARAMETER_NAMES),
    "cross_section_y_range": (1e-4, 1e4),
    "potential_y_range": (-60.0, 60.0),
    "plot_colors": {
        "live_lrom": "#162033",
        "central_lrom": "#2563eb",
        "central_fom_evaluation": "#dc2626",
    },
    "initial_visibility": {
        "central_lrom": True,
        "central_fom_evaluation": True,
        "potential_inset": True,
    },
}
_NAMED_OPTION_MAPS = (
    "slider_ranges",
    "slider_steps",
    "slider_formats",
    "parameter_labels",
    "parameter_groups",
)
_MODEL_CACHE_KEYS = (
    "partial_waves",
    "ell",
    "spin",
    "ldots",
    "energy_scales",
    "feature_shape",
    "feature_radii",
    "feature_ldots",
    "feature_energy_scales",
    "centers",
    "scales",
    "matrices",
    "vectors",
    "constants",
    "asymptotic_values",
    "asymptotic_derivatives",
    "hminus",
    "hplus",
    "hminus_derivative",
    "hplus_derivative",
    "s0",
    "plus_channel_indices",
    "plus_ell_indices",
    "minus_channel_indices",
    "minus_ell_indices",
    "angles_degrees",
)

_HTML_DATA_MARKER = "__LROM_HTML_DATA__"
_BASE_CSS = """
:root {
  color-scheme: light;
  --ink: #111827;
  --muted: #6b7280;
  --line: #d1d5db;
  --blue: #1d4ed8;
  --green: #047857;
  --paper: #ffffff;
  --soft: #f8fafc;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--paper);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); }
button, input { font: inherit; }
#explorer { width: min(1500px, calc(100vw - 44px)); margin: 24px auto 42px; }
.explorer-header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 16px;
  margin-bottom: 20px;
}
.explorer-header h1 { margin: 0; font-size: 34px; font-weight: 760; letter-spacing: 0; }
.explorer-header p { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
.integrated-card {
  min-width: 280px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--soft);
}
.integrated-card span { display: block; color: var(--muted); font-size: 15px; margin-bottom: 5px; }
.integrated-card strong { display: block; font-size: 28px; font-variant-numeric: tabular-nums; }
.explorer-grid {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 22px;
  align-items: start;
}
.control-panel, .plot-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}
.control-panel { padding: 14px 14px 10px; }
.parameter-group { margin: 7px 0 0; padding: 0; border: 0; }
.parameter-group legend {
  padding: 12px 0 3px;
  width: 100%;
  border-top: 1px solid #dbe4ef;
  color: #475569;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.parameter-group:first-child { margin-top: 0; }
.parameter-group:first-child legend { border-top: none; padding-top: 3px; }
.slider-row { padding: 10px 0 12px; border-bottom: 1px solid #eef2f7; }
.slider-row:last-child { border-bottom: none; }
.slider-label {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 7px;
  font-weight: 650;
  font-size: 14px;
}
.slider-label output {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: #1f2937;
}
.slider-row input[type="range"] { width: 100%; accent-color: var(--blue); }
.slider-bounds {
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.view-controls {
  margin-top: 10px;
  padding-top: 12px;
  border-top: 1px solid #dbe4ef;
}
.view-controls label { display: block; margin: .3rem 0; font-size: 14px; font-weight: 650; }
.button-row { display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 0 4px; }
button {
  border: 1px solid #cbd5e1;
  background: #fff;
  color: var(--ink);
  border-radius: 7px;
  padding: 8px 11px;
  font-weight: 650;
  cursor: pointer;
}
button:hover { background: #f1f5f9; }
.status, .error { min-height: 1.3rem; margin-top: .55rem; font-size: .9rem; }
.status { color: #9a3412; }
.error { color: #b91c1c; }
.plot-panel { min-width: 0; padding: 12px; }
svg#cross-section-plot {
  display: block;
  width: 100%;
  height: min(74vh, 760px);
  min-height: 520px;
}
.axis text { fill: #374151; font-size: 13px; }
.axis path, .axis line, .grid line { stroke: #cbd5e1; stroke-width: 1; }
.grid line { opacity: 0.7; }
.curve { fill: none; stroke-width: 4.6; }
.reference { fill: none; stroke-width: 3; stroke-dasharray: 7 5; opacity: 0.75; }
.legend-box, .inset-box { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; opacity: 0.96; }
.scale-controls {
  display: grid;
  grid-template-columns: minmax(170px, 0.7fr) minmax(220px, 1fr) minmax(220px, 1fr);
  align-items: center;
  gap: 18px;
  margin: 5px 8px 10px;
  padding: 12px 14px;
  border-top: 1px solid #dbe4ef;
  background: var(--soft);
}
.scale-title { font-size: 14px; font-weight: 750; color: var(--ink); }
.scale-control .row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 3px;
}
.scale-control label { font-size: 13px; font-weight: 650; }
.scale-control output {
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}
.scale-controls input[type="range"] { width: 100%; accent-color: #111827; }
.caption { color: var(--muted); font-size: 13px; padding: 0 8px 8px; }
.explorer-footer { padding: 16px 2px 0; color: var(--muted); font-size: 13px; }
[hidden] { display: none !important; }
@media (max-width: 980px) {
  .explorer-header, .explorer-grid { display: block; }
  .integrated-card { min-width: 0; margin-top: 16px; }
  .control-panel { margin-bottom: 18px; }
  .scale-controls { grid-template-columns: 1fr; gap: 10px; }
  svg#cross-section-plot { min-height: 420px; }
}
"""
_NUMERICS_JS = r"""
"use strict";
(() => {
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const subtract = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const multiply = (a, b) => [
    a[0] * b[0] - a[1] * b[1],
    a[0] * b[1] + a[1] * b[0],
  ];
  const scale = (a, value) => [a[0] * value, a[1] * value];
  const divide = (a, b) => {
    const denominator = b[0] * b[0] + b[1] * b[1];
    if (!Number.isFinite(denominator) || denominator <= 1e-30) {
      throw new Error("singular complex division");
    }
    return [
      (a[0] * b[0] + a[1] * b[1]) / denominator,
      (a[1] * b[0] - a[0] * b[1]) / denominator,
    ];
  };
  const magnitudeSquared = value => value[0] ** 2 + value[1] ** 2;
  const finiteComplex = value => (
    Number.isFinite(value[0]) && Number.isFinite(value[1])
  );

  function solveComplex(matrix, rhs) {
    const size = rhs.length;
    const a = matrix.map(row => row.map(value => [...value]));
    const b = rhs.map(value => [...value]);
    for (let column = 0; column < size; column += 1) {
      let pivot = column;
      for (let row = column + 1; row < size; row += 1) {
        if (magnitudeSquared(a[row][column]) > magnitudeSquared(a[pivot][column])) {
          pivot = row;
        }
      }
      if (magnitudeSquared(a[pivot][column]) <= 1e-28) {
        throw new Error("singular RF system");
      }
      if (pivot !== column) {
        [a[pivot], a[column]] = [a[column], a[pivot]];
        [b[pivot], b[column]] = [b[column], b[pivot]];
      }
      for (let row = column + 1; row < size; row += 1) {
        const factor = divide(a[row][column], a[column][column]);
        for (let index = column; index < size; index += 1) {
          a[row][index] = subtract(
            a[row][index],
            multiply(factor, a[column][index]),
          );
        }
        b[row] = subtract(b[row], multiply(factor, b[column]));
      }
    }
    const result = Array.from({length: size}, () => [0, 0]);
    for (let row = size - 1; row >= 0; row -= 1) {
      let value = [...b[row]];
      for (let column = row + 1; column < size; column += 1) {
        value = subtract(value, multiply(a[row][column], result[column]));
      }
      result[row] = divide(value, a[row][row]);
      if (!finiteComplex(result[row])) {
        throw new Error("non-finite RF solution");
      }
    }
    return result;
  }

  function potentialTerms(htmlData, parameters) {
    const model = htmlData.model;
    const [vv, wv, wd, vso, rv, rd, rso, av, ad, aso] = parameters;
    if (![av, ad, aso].every(value => Number.isFinite(value) && value > 0)) {
      throw new Error("diffuseness parameters must be positive");
    }
    const radii = model.potential_radii_fm;
    const realVolume = [];
    const imaginaryVolume = [];
    const imaginarySurface = [];
    const spinOrbit = [];
    for (const radius of radii) {
      const volumeExponent = Math.max(-700, Math.min(700, (radius - rv) / av));
      const volume = 1 / (1 + Math.exp(volumeExponent));
      const surfaceExponent = Math.max(-700, Math.min(700, (radius - rd) / ad));
      const surfaceExp = Math.exp(surfaceExponent);
      const surfacePrime = -(surfaceExp / ad) / (1 + surfaceExp) ** 2;
      const spinExponent = Math.max(-700, Math.min(700, (radius - rso) / aso));
      const spinExp = Math.exp(spinExponent);
      const spinPrime = -(spinExp / aso) / (1 + spinExp) ** 2;
      realVolume.push(-vv * volume);
      imaginaryVolume.push(-wv * volume);
      imaginarySurface.push(4 * ad * wd * surfacePrime);
      spinOrbit.push(vso / model.mass_pion_squared * spinPrime / radius);
    }
    return {
      radii_fm: [...radii],
      real_volume: realVolume,
      imaginary_volume: imaginaryVolume,
      imaginary_surface: imaginarySurface,
      spin_orbit: spinOrbit,
    };
  }

  function effectiveInteractionFeatures(model, parameters) {
    const [vv, wv, wd, vso, rv, rd, rso, av, ad, aso] = parameters;
    if (![av, ad, aso].every(value => Number.isFinite(value) && value > 0)) {
      throw new Error("diffuseness parameters must be positive");
    }
    const raw = [];
    for (let index = 0; index < model.feature_radii.length; index += 1) {
      const radius = model.feature_radii[index];
      const volumeExponent = Math.max(-700, Math.min(700, (radius - rv) / av));
      const volume = 1 / (1 + Math.exp(volumeExponent));
      const surfaceExponent = Math.max(-700, Math.min(700, (radius - rd) / ad));
      const surfaceExp = Math.exp(surfaceExponent);
      const surfacePrime = -(surfaceExp / ad) / (1 + surfaceExp) ** 2;
      const spinExponent = Math.max(-700, Math.min(700, (radius - rso) / aso));
      const spinExp = Math.exp(spinExponent);
      const spinPrime = -(spinExp / aso) / (1 + spinExp) ** 2;
      const real = -vv * volume + (
        model.feature_ldots[index] * vso / model.mass_pion_squared
        * spinPrime / radius
      );
      const imaginary = -wv * volume + 4 * ad * wd * surfacePrime;
      raw.push(scale([real, imaginary], 1 / model.feature_energy_scales[index]));
    }
    const [, predictorCount] = model.feature_shape;
    return model.channels.map((_, channelIndex) => (
      Array.from({length: predictorCount}, (_unused, predictorIndex) => {
        const offset = channelIndex * predictorCount + predictorIndex;
        return scale(
          subtract(raw[offset], model.centers[channelIndex][predictorIndex]),
          1 / model.scales[channelIndex][predictorIndex],
        );
      })
    ));
  }

  function rfSystem(model, features, channelIndex) {
    const basisSize = model.constants[channelIndex].length;
    const matrix = Array.from({length: basisSize}, (_unused, row) => (
      Array.from({length: basisSize}, (_unusedAgain, column) => (
        [row === column ? 1 : 0, 0]
      ))
    ));
    const rhs = model.constants[channelIndex].map(value => [...value]);
    for (let predictor = 0; predictor < features.length; predictor += 1) {
      for (let row = 0; row < basisSize; row += 1) {
        rhs[row] = add(
          rhs[row],
          multiply(features[predictor], model.vectors[channelIndex][predictor][row]),
        );
        for (let column = 0; column < basisSize; column += 1) {
          matrix[row][column] = add(
            matrix[row][column],
            multiply(
              features[predictor],
              model.matrices[channelIndex][predictor][row][column],
            ),
          );
        }
      }
    }
    return {matrix, rhs};
  }

  function smatrixFromCoordinates(model, coordinates) {
    const packed = coordinates.map((channelCoordinates, channelIndex) => {
      const expansion = [[1, 0], ...channelCoordinates];
      let phi = [0, 0];
      let phiPrime = [0, 0];
      for (let index = 0; index < expansion.length; index += 1) {
        phi = add(
          phi,
          multiply(expansion[index], model.asymptotic_values[channelIndex][index]),
        );
        phiPrime = add(
          phiPrime,
          multiply(
            expansion[index],
            model.asymptotic_derivatives[channelIndex][index],
          ),
        );
      }
      const s0 = model.s0[channelIndex];
      const rMatrix = divide(phi, scale(phiPrime, s0));
      const numerator = subtract(
        model.hminus[channelIndex],
        scale(
          multiply(rMatrix, model.hminus_derivative[channelIndex]),
          s0,
        ),
      );
      const denominator = subtract(
        model.hplus[channelIndex],
        scale(
          multiply(rMatrix, model.hplus_derivative[channelIndex]),
          s0,
        ),
      );
      return divide(numerator, denominator);
    });
    const count = model.partial_waves.length;
    const splus = Array.from({length: count}, () => [0, 0]);
    const sminus = Array.from({length: count}, () => [0, 0]);
    model.plus_channel_indices.forEach((channelIndex, index) => {
      splus[model.plus_ell_indices[index]] = packed[channelIndex];
    });
    model.minus_channel_indices.forEach((channelIndex, index) => {
      sminus[model.minus_ell_indices[index]] = packed[channelIndex];
    });
    return {splus, sminus};
  }

  function neutralCrossSection(model, smatrix) {
    const result = [];
    const divisor = [0, 2 * model.momentum_k];
    for (let angleIndex = 0; angleIndex < model.angles_degrees.length; angleIndex += 1) {
      let amplitudeA = [0, 0];
      let amplitudeB = [0, 0];
      for (let ell = 0; ell < model.partial_waves.length; ell += 1) {
        const plus = subtract(smatrix.splus[ell], [1, 0]);
        const minus = subtract(smatrix.sminus[ell], [1, 0]);
        const weighted = add(scale(plus, ell + 1), scale(minus, ell));
        amplitudeA = add(
          amplitudeA,
          scale(
            divide(weighted, divisor),
            model.p_l_costheta[ell][angleIndex],
          ),
        );
        amplitudeB = add(
          amplitudeB,
          scale(
            divide(subtract(smatrix.splus[ell], smatrix.sminus[ell]), divisor),
            model.p_1_l_costheta[ell][angleIndex],
          ),
        );
      }
      result.push(10 * (magnitudeSquared(amplitudeA) + magnitudeSquared(amplitudeB)));
    }
    return result;
  }

  function integratedCrossSection(angles, values) {
    let result = 0;
    for (let index = 1; index < angles.length; index += 1) {
      const leftTheta = angles[index - 1] * Math.PI / 180;
      const rightTheta = angles[index] * Math.PI / 180;
      const left = 2 * Math.PI * values[index - 1] * Math.sin(leftTheta);
      const right = 2 * Math.PI * values[index] * Math.sin(rightTheta);
      result += 0.5 * (left + right) * (rightTheta - leftTheta);
    }
    return result;
  }

  function predict(htmlData, parameters) {
    if (parameters.length !== 10 || !parameters.every(Number.isFinite)) {
      throw new Error("prediction requires ten finite parameters");
    }
    const model = htmlData.model;
    const features = effectiveInteractionFeatures(model, parameters);
    const coordinates = model.channels.map((_channel, channelIndex) => {
      const system = rfSystem(model, features[channelIndex], channelIndex);
      return solveComplex(system.matrix, system.rhs);
    });
    const smatrix = smatrixFromCoordinates(model, coordinates);
    const crossSection = neutralCrossSection(model, smatrix);
    return {
      cross_section: crossSection,
      integrated_cross_section: integratedCrossSection(
        htmlData.science.angles_degrees,
        crossSection,
      ),
    };
  }

  globalThis.LromNumerics = Object.freeze({
    predict,
    potentialTerms,
    integratedCrossSection,
  });
})();
"""
_BOOTSTRAP_JS = """
"use strict";
const LROM_HTML_DATA = Object.freeze(JSON.parse(
  document.getElementById("lrom-html-data").textContent
));
"""
_INTERFACE_HTML = """
<header class="explorer-header">
  <div><h1 id="explorer-title"></h1><p id="explorer-subtitle"></p></div>
  <div class="integrated-card">
    <span>Angle-integrated elastic cross section</span>
    <strong id="integrated-value">&#8212;</strong>
  </div>
</header>
<div class="explorer-grid">
  <aside class="control-panel" aria-label="Optical-potential controls">
    <div id="parameter-controls"></div>
    <div class="view-controls">
      <label><input type="checkbox" data-view="central-lrom"> Central LROM</label>
      <label id="central-fom-control"><input type="checkbox" data-view="central-fom-evaluation"> Central FOM Evaluation</label>
      <label><input type="checkbox" data-view="potential-inset"> Potential figure</label>
    </div>
    <div class="button-row">
      <button type="button" id="reset-center">Reset Center</button>
      <button type="button" id="random-training">Random Training Point</button>
    </div>
    <div id="extrapolation-status" class="status" role="status"></div>
    <div id="prediction-error" class="error" role="alert"></div>
  </aside>
  <section class="plot-panel" aria-label="Cross-section plot">
    <svg id="cross-section-plot" viewBox="0 0 1040 680" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Differential cross section against scattering angle"></svg>
    <div class="scale-controls">
      <div class="scale-title">Cross-section plot scale</div>
      <div class="scale-control">
        <div class="row"><label for="xs-min-exponent">Y-axis minimum</label><output id="xs-min-value"></output></div>
        <input id="xs-min-exponent" type="range" step="1">
      </div>
      <div class="scale-control">
        <div class="row"><label for="xs-max-exponent">Y-axis maximum</label><output id="xs-max-value"></output></div>
        <input id="xs-max-exponent" type="range" step="1">
      </div>
    </div>
    <div class="scale-controls" id="potential-scale-controls">
      <div class="scale-title">Potential inset scale</div>
      <div class="scale-control">
        <div class="row"><label for="potential-minimum">Y-axis minimum</label><output id="potential-min-value"></output></div>
        <input id="potential-minimum" type="range" step="10">
      </div>
      <div class="scale-control">
        <div class="row"><label for="potential-maximum">Y-axis maximum</label><output id="potential-max-value"></output></div>
        <input id="potential-maximum" type="range" step="10">
      </div>
    </div>
    <div class="caption">Y axis is logarithmic. The inset shows the scaled local optical-potential terms.</div>
  </section>
</div>
<footer id="explorer-footer" class="explorer-footer"></footer>
<template id="parameter-slider-template">
  <div class="slider-row">
    <label class="slider-label"><span></span><output></output></label>
    <input type="range">
    <div class="slider-bounds"><span></span><span></span></div>
  </div>
</template>
"""
_INTERFACE_JS = r"""
"use strict";
(() => {
  const data = LROM_HTML_DATA;
  const byId = id => document.getElementById(id);
  const svgNS = "http://www.w3.org/2000/svg";
  const colors = data.options.plot_colors;

  const title = byId("explorer-title");
  const subtitle = byId("explorer-subtitle");
  const footer = byId("explorer-footer");
  title.textContent = data.options.title;
  subtitle.textContent = data.options.subtitle || (
    `${data.science.reaction_label} at ${data.science.lab_energy_mev} MeV, `
    + `Nφ=${data.science.basis_size}, Np=${data.science.predictor_count}, `
    + `lmax=${data.science.l_max}`
  );
  footer.textContent = data.options.footer_note || (
    "The live curve is evaluated locally in this file. Radius is shown in fm."
  );

  const controls = byId("parameter-controls");
  const sliderTemplate = byId("parameter-slider-template");
  const sliders = new Map();
  const grouped = new Map();
  for (const parameter of data.parameters) {
    if (!grouped.has(parameter.group)) grouped.set(parameter.group, []);
    grouped.get(parameter.group).push(parameter);
  }
  for (const [group, parameters] of grouped) {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "parameter-group";
    const legend = document.createElement("legend");
    legend.textContent = group;
    fieldset.appendChild(legend);
    for (const parameter of parameters) {
      const row = sliderTemplate.content.firstElementChild.cloneNode(true);
      const input = row.querySelector("input");
      const label = row.querySelector("label");
      const labelText = row.querySelector("label span");
      const output = row.querySelector("output");
      const bounds = row.querySelectorAll(".slider-bounds span");
      const inputId = `parameter-${parameter.name}`;
      input.id = inputId;
      input.min = parameter.slider_min;
      input.max = parameter.slider_max;
      input.step = parameter.step;
      input.value = parameter.central;
      label.htmlFor = inputId;
      labelText.textContent = parameter.label;
      output.htmlFor = inputId;
      bounds[0].textContent = Number(parameter.slider_min).toFixed(parameter.precision);
      bounds[1].textContent = Number(parameter.slider_max).toFixed(parameter.precision);
      sliders.set(parameter.name, {input, output, parameter});
      fieldset.appendChild(row);
    }
    controls.appendChild(fieldset);
  }

  const centralToggle = document.querySelector('[data-view="central-lrom"]');
  const fomToggle = document.querySelector('[data-view="central-fom-evaluation"]');
  const potentialToggle = document.querySelector('[data-view="potential-inset"]');
  const fomControl = byId("central-fom-control");
  centralToggle.checked = data.options.initial_visibility.central_lrom;
  fomToggle.checked = data.options.initial_visibility.central_fom_evaluation;
  potentialToggle.checked = data.options.initial_visibility.potential_inset;
  if (data.references.central_fom_evaluation === null) {
    fomToggle.checked = false;
    fomToggle.disabled = true;
    fomControl.hidden = true;
  }

  const svg = byId("cross-section-plot");
  const xsMinInput = byId("xs-min-exponent");
  const xsMaxInput = byId("xs-max-exponent");
  const xsMinValue = byId("xs-min-value");
  const xsMaxValue = byId("xs-max-value");
  const potentialMinInput = byId("potential-minimum");
  const potentialMaxInput = byId("potential-maximum");
  const potentialMinValue = byId("potential-min-value");
  const potentialMaxValue = byId("potential-max-value");
  const potentialScaleControls = byId("potential-scale-controls");

  const [initialLow, initialHigh] = data.options.cross_section_y_range;
  const initialMinExponent = Math.round(Math.log10(initialLow));
  const initialMaxExponent = Math.round(Math.log10(initialHigh));
  const exponentFloor = Math.min(-8, initialMinExponent);
  const exponentCeiling = Math.max(6, initialMaxExponent);
  xsMinInput.min = exponentFloor;
  xsMinInput.max = exponentCeiling - 1;
  xsMaxInput.min = exponentFloor + 1;
  xsMaxInput.max = exponentCeiling;
  xsMinInput.value = initialMinExponent;
  xsMaxInput.value = initialMaxExponent;

  const [initialPotentialLow, initialPotentialHigh] = data.options.potential_y_range;
  const potentialFloor = Math.min(-200, Math.floor(initialPotentialLow / 10) * 10);
  const potentialCeiling = Math.max(200, Math.ceil(initialPotentialHigh / 10) * 10);
  potentialMinInput.min = potentialFloor;
  potentialMinInput.max = potentialCeiling - 10;
  potentialMaxInput.min = potentialFloor + 10;
  potentialMaxInput.max = potentialCeiling;
  potentialMinInput.value = initialPotentialLow;
  potentialMaxInput.value = initialPotentialHigh;

  let crossSectionYMin = initialLow;
  let crossSectionYMax = initialHigh;
  let potentialYMin = initialPotentialLow;
  let potentialYMax = initialPotentialHigh;

  const SUPERSCRIPTS = {
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
    "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
  };
  const superscriptInteger = value => String(value)
    .split("")
    .map(character => SUPERSCRIPTS[character] || character)
    .join("");
  const reactionLabel = data.science.reaction_label.replace(
    /^\d+/,
    match => superscriptInteger(match),
  );

  function updateScaleLabels() {
    xsMinValue.textContent = `10${superscriptInteger(Number(xsMinInput.value))}`;
    xsMaxValue.textContent = `10${superscriptInteger(Number(xsMaxInput.value))}`;
    potentialMinValue.textContent = `${Number(potentialMinInput.value).toFixed(0)} MeV`;
    potentialMaxValue.textContent = `${Number(potentialMaxInput.value).toFixed(0)} MeV`;
  }

  function enforceCrossSectionScale(changed) {
    let minimum = Number(xsMinInput.value);
    let maximum = Number(xsMaxInput.value);
    if (minimum >= maximum) {
      if (changed === "minimum") {
        maximum = Math.min(exponentCeiling, minimum + 1);
        if (maximum <= minimum) minimum = maximum - 1;
      } else {
        minimum = Math.max(exponentFloor, maximum - 1);
        if (minimum >= maximum) maximum = minimum + 1;
      }
      xsMinInput.value = minimum;
      xsMaxInput.value = maximum;
    }
    crossSectionYMin = Math.pow(10, Number(xsMinInput.value));
    crossSectionYMax = Math.pow(10, Number(xsMaxInput.value));
    updateScaleLabels();
  }

  function enforcePotentialScale(changed) {
    let minimum = Number(potentialMinInput.value);
    let maximum = Number(potentialMaxInput.value);
    if (minimum >= maximum) {
      if (changed === "minimum") {
        maximum = Math.min(potentialCeiling, minimum + 10);
        if (maximum <= minimum) minimum = maximum - 10;
      } else {
        minimum = Math.max(potentialFloor, maximum - 10);
        if (minimum >= maximum) maximum = minimum + 10;
      }
      potentialMinInput.value = minimum;
      potentialMaxInput.value = maximum;
    }
    potentialYMin = Number(potentialMinInput.value);
    potentialYMax = Number(potentialMaxInput.value);
    updateScaleLabels();
  }

  const WIDTH = 1040;
  const HEIGHT = 680;
  const MARGIN = {left: 88, right: 24, top: 30, bottom: 76};
  const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
  const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;
  const POTENTIAL_SERIES = [
    {key: "real_volume", label: "real volume", color: "#2563eb", dash: "", factor: 1},
    {key: "imaginary_volume", label: "imag volume ×3", color: "#dc2626", dash: "7 4", factor: 3},
    {key: "imaginary_surface", label: "imag surface ×3", color: "#f97316", dash: "3 3", factor: 3},
    {key: "spin_orbit", label: "spin-orbit ×25", color: "#16a34a", dash: "8 3 2 3", factor: 25},
  ];

  function clearSvg() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }
  function element(name, attributes) {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attributes)) {
      if (value === null || value === undefined || value === "") continue;
      node.setAttribute(key, value);
    }
    svg.appendChild(node);
    return node;
  }
  function line(x1, y1, x2, y2, className, stroke, width) {
    return element("line", {
      x1, y1, x2, y2,
      class: className || null,
      stroke: stroke || "#cbd5e1",
      "stroke-width": width === undefined ? 1 : width,
    });
  }
  function text(x, y, content, anchor, size, rotate) {
    const node = element("text", {
      x, y,
      "text-anchor": anchor || "middle",
      "font-size": size === undefined ? 15 : size,
      fill: "#374151",
      transform: rotate ? `rotate(${rotate}, ${x}, ${y})` : null,
    });
    node.textContent = content;
    return node;
  }
  function powerOfTenText(x, y, exponent, size) {
    const node = element("text", {
      x, y,
      "text-anchor": "end",
      "font-size": size,
      fill: "#374151",
    });
    const base = document.createElementNS(svgNS, "tspan");
    base.textContent = "10";
    node.appendChild(base);
    if (exponent !== 0) {
      const power = document.createElementNS(svgNS, "tspan");
      power.setAttribute("baseline-shift", "super");
      power.setAttribute("font-size", Math.round(size * 0.72));
      power.textContent = String(exponent);
      node.appendChild(power);
    }
    return node;
  }
  function rect(x, y, width, height, className, rx) {
    return element("rect", {
      x, y, width, height,
      rx: rx === undefined ? 6 : rx,
      class: className || null,
    });
  }

  const angleToX = angle => MARGIN.left + angle / 180 * PLOT_WIDTH;
  function valueToY(value) {
    if (!Number.isFinite(value) || value <= 0) return null;
    if (value < crossSectionYMin || value > crossSectionYMax) return null;
    const span = Math.log10(crossSectionYMax) - Math.log10(crossSectionYMin);
    return MARGIN.top + (Math.log10(crossSectionYMax) - Math.log10(value)) / span * PLOT_HEIGHT;
  }
  function pathFor(angles, values) {
    let commands = "";
    let drawing = false;
    for (let index = 0; index < values.length; index += 1) {
      const y = valueToY(values[index]);
      if (y === null) {
        drawing = false;
        continue;
      }
      commands += `${drawing ? "L" : "M"}${angleToX(angles[index]).toFixed(2)},${y.toFixed(2)} `;
      drawing = true;
    }
    return commands.trim();
  }
  function curve(angles, values, className, stroke) {
    const commands = pathFor(angles, values);
    if (!commands) return null;
    return element("path", {
      d: commands,
      class: className,
      fill: "none",
      stroke,
    });
  }

  function drawFrame() {
    for (const angle of [0, 30, 60, 90, 120, 150, 180]) {
      const x = angleToX(angle);
      line(x, MARGIN.top, x, MARGIN.top + PLOT_HEIGHT, "grid");
      text(x, HEIGHT - 40, String(angle), "middle", 18);
    }
    const lowest = Math.ceil(Math.log10(crossSectionYMin));
    const highest = Math.floor(Math.log10(crossSectionYMax));
    for (let exponent = lowest; exponent <= highest; exponent += 1) {
      const y = valueToY(Math.pow(10, exponent));
      if (y === null) continue;
      line(MARGIN.left, y, MARGIN.left + PLOT_WIDTH, y, "grid");
      powerOfTenText(MARGIN.left - 12, y + 6, exponent, 17);
    }
    line(MARGIN.left, MARGIN.top + PLOT_HEIGHT, MARGIN.left + PLOT_WIDTH, MARGIN.top + PLOT_HEIGHT, "axis", "#111827", 1.2);
    line(MARGIN.left, MARGIN.top, MARGIN.left, MARGIN.top + PLOT_HEIGHT, "axis", "#111827", 1.2);
    text(MARGIN.left + PLOT_WIDTH / 2, HEIGHT - 10, "θ (deg)", "middle", 22);
    text(22, MARGIN.top + PLOT_HEIGHT / 2, "dσ/dΩ (mb/sr)", "middle", 22, -90);
    const annotation = text(
      MARGIN.left + 20,
      MARGIN.top + PLOT_HEIGHT - 24,
      `${reactionLabel} at ${data.science.lab_energy_mev} MeV`,
      "start",
      28,
    );
    annotation.setAttribute("font-weight", "700");
    annotation.setAttribute("fill", "#111827");
  }

  function drawMarkers(angles, values, color) {
    for (let index = 0; index < values.length; index += 3) {
      const y = valueToY(values[index]);
      if (y === null) continue;
      element("circle", {
        cx: angleToX(angles[index]).toFixed(2),
        cy: y.toFixed(2),
        r: 3.2,
        fill: "none",
        stroke: color,
        "stroke-width": 1.4,
      });
    }
  }

  function drawLegend(entries) {
    if (!entries.length) return;
    const x = MARGIN.left + 18;
    const y = MARGIN.top + 18;
    const height = 22 + entries.length * 29;
    rect(x, y, 290, height, "legend-box");
    entries.forEach((entry, index) => {
      const row = y + 25 + index * 29;
      if (entry.marker) {
        element("circle", {
          cx: x + 37, cy: row, r: 4,
          fill: "none", stroke: entry.color, "stroke-width": 1.6,
        });
      } else {
        const sample = line(x + 14, row, x + 60, row, "", entry.color, entry.width);
        if (entry.dash) sample.setAttribute("stroke-dasharray", entry.dash);
      }
      text(x + 71, row + 6, entry.label, "start", 15);
    });
  }

  function drawPotentialInset(potential) {
    const inset = {x: 615, y: 48, width: 385, height: 315};
    rect(inset.x, inset.y, inset.width, inset.height, "inset-box");
    text(inset.x + 17, inset.y + 27, "Scaled optical-potential terms", "start", 17);
    const frame = {
      left: inset.x + 58,
      right: inset.x + inset.width - 17,
      top: inset.y + 44,
      bottom: inset.y + 215,
    };
    const radii = potential.radii_fm;
    const maximumRadius = radii[radii.length - 1];
    const radiusToX = radius => frame.left + radius / maximumRadius * (frame.right - frame.left);
    const termToY = value => {
      if (!Number.isFinite(value) || value < potentialYMin || value > potentialYMax) return null;
      return frame.top + (potentialYMax - value) / (potentialYMax - potentialYMin) * (frame.bottom - frame.top);
    };
    for (let tick = 0; tick <= maximumRadius + 1e-9; tick += maximumRadius / 5) {
      const x = radiusToX(tick);
      line(x, frame.top, x, frame.bottom, "", "#e2e8f0", 0.8);
      text(x, frame.bottom + 18, tick.toFixed(0), "middle", 12);
    }
    for (let step = 0; step <= 4; step += 1) {
      const tick = potentialYMin + step * (potentialYMax - potentialYMin) / 4;
      const y = termToY(tick);
      if (y === null) continue;
      line(frame.left, y, frame.right, y, "", "#e2e8f0", 0.8);
      text(frame.left - 8, y + 4, Math.abs(tick) >= 10 ? tick.toFixed(0) : tick.toFixed(1), "end", 12);
    }
    line(frame.left, frame.bottom, frame.right, frame.bottom, "", "#475569", 1);
    line(frame.left, frame.top, frame.left, frame.bottom, "", "#475569", 1);
    text((frame.left + frame.right) / 2, frame.bottom + 37, "r (fm)", "middle", 14);
    text(inset.x + 17, (frame.top + frame.bottom) / 2, "scaled U term (MeV)", "middle", 13, -90);

    for (const series of POTENTIAL_SERIES) {
      const values = potential[series.key];
      let commands = "";
      let drawing = false;
      for (let index = 0; index < values.length; index += 1) {
        const y = termToY(values[index] * series.factor);
        if (y === null) {
          drawing = false;
          continue;
        }
        commands += `${drawing ? "L" : "M"}${radiusToX(radii[index]).toFixed(2)},${y.toFixed(2)} `;
        drawing = true;
      }
      if (!commands) continue;
      const path = element("path", {
        d: commands.trim(),
        fill: "none",
        stroke: series.color,
        "stroke-width": 2.5,
      });
      if (series.dash) path.setAttribute("stroke-dasharray", series.dash);
    }
    POTENTIAL_SERIES.forEach((series, index) => {
      const column = index % 2;
      const row = Math.floor(index / 2);
      const x = inset.x + 18 + column * 181;
      const y = inset.y + 263 + row * 27;
      const sample = line(x, y, x + 31, y, "", series.color, 2.6);
      if (series.dash) sample.setAttribute("stroke-dasharray", series.dash);
      text(x + 39, y + 5, series.label, "start", 12);
    });
  }

  let lastPrediction = null;
  let lastPotential = null;
  let scheduled = false;

  function parameterVector() {
    const values = Array(10);
    for (const {input, output, parameter} of sliders.values()) {
      const value = Number(input.value);
      if (!Number.isFinite(value)) throw new Error(`${parameter.label} is not finite`);
      values[parameter.model_index] = value;
      output.textContent = value.toFixed(parameter.precision);
    }
    return values;
  }

  function updateDomainStatus(values) {
    const outside = data.parameters.filter(parameter => (
      values[parameter.model_index] < parameter.training_min
      || values[parameter.model_index] > parameter.training_max
    ));
    byId("extrapolation-status").textContent = outside.length
      ? `Outside the training domain: ${outside.map(item => item.label).join(", ")}`
      : "";
  }

  function renderPlots() {
    if (lastPrediction === null || lastPotential === null) return;
    const angles = data.science.angles_degrees;
    const reference = data.references.central_lrom;
    const fom = data.references.central_fom_evaluation;
    const showPotential = potentialToggle.checked;
    potentialScaleControls.hidden = !showPotential;
    clearSvg();
    drawFrame();
    const legend = [];
    if (centralToggle.checked) {
      curve(reference.angles_degrees, reference.cross_section, "reference", colors.central_lrom);
      legend.push({
        label: "Central LROM",
        color: colors.central_lrom,
        width: 3,
        dash: "7 5",
      });
    }
    if (fom !== null && fomToggle.checked) {
      drawMarkers(fom.angles_degrees, fom.cross_section, colors.central_fom_evaluation);
      legend.push({
        label: "Central FOM Evaluation",
        color: colors.central_fom_evaluation,
        marker: true,
      });
    }
    curve(angles, lastPrediction.cross_section, "curve", colors.live_lrom);
    legend.unshift({label: "Live LROM", color: colors.live_lrom, width: 4.6});
    drawLegend(legend);
    if (showPotential) drawPotentialInset(lastPotential);
  }

  function evaluate() {
    scheduled = false;
    try {
      const values = parameterVector();
      updateDomainStatus(values);
      const prediction = LromNumerics.predict(data, values);
      const potential = LromNumerics.potentialTerms(data, values);
      lastPrediction = prediction;
      lastPotential = potential;
      byId("integrated-value").textContent = (
        `${prediction.integrated_cross_section.toExponential(3)} mb`
      );
      byId("prediction-error").textContent = "";
      renderPlots();
    } catch (error) {
      byId("prediction-error").textContent = `Prediction unavailable: ${error.message}`;
    }
  }

  function scheduleEvaluation() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(evaluate);
  }

  for (const {input} of sliders.values()) input.addEventListener("input", scheduleEvaluation);
  for (const toggle of [centralToggle, fomToggle, potentialToggle]) {
    toggle.addEventListener("change", renderPlots);
  }
  xsMinInput.addEventListener("input", () => {
    enforceCrossSectionScale("minimum");
    renderPlots();
  });
  xsMaxInput.addEventListener("input", () => {
    enforceCrossSectionScale("maximum");
    renderPlots();
  });
  potentialMinInput.addEventListener("input", () => {
    enforcePotentialScale("minimum");
    renderPlots();
  });
  potentialMaxInput.addEventListener("input", () => {
    enforcePotentialScale("maximum");
    renderPlots();
  });
  byId("reset-center").addEventListener("click", () => {
    for (const {input, parameter} of sliders.values()) input.value = parameter.central;
    scheduleEvaluation();
  });
  byId("random-training").addEventListener("click", () => {
    for (const {input, parameter} of sliders.values()) {
      input.value = parameter.training_min + Math.random() * (
        parameter.training_max - parameter.training_min
      );
    }
    scheduleEvaluation();
  });

  enforceCrossSectionScale("maximum");
  enforcePotentialScale("maximum");
  evaluate();
  globalThis.LROM_EXPLORER_READY = true;
})();
"""
_HTML_TEMPLATE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LROM Cross Section Explorer</title>
<style>{_BASE_CSS}</style>
</head>
<body>
<main id="explorer">{_INTERFACE_HTML}</main>
<script id="lrom-numerics">{_NUMERICS_JS}</script>
<script id="lrom-html-data" type="application/json">{_HTML_DATA_MARKER}</script>
<script>{_BOOTSTRAP_JS}</script>
<script>{_INTERFACE_JS}</script>
</body>
</html>
"""


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _finite_range(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a (minimum, maximum) pair")
    lower = _finite_float(value[0], name=f"{name} minimum")
    upper = _finite_float(value[1], name=f"{name} maximum")
    if lower >= upper:
        raise ValueError(f"{name} must be increasing")
    return lower, upper


def _validated_options(options: Mapping[str, object] | None) -> dict[str, object]:
    supplied = {} if options is None else dict(options)
    unknown = sorted(set(supplied) - set(_DEFAULT_OPTIONS))
    if unknown:
        raise ValueError(f"unknown explorer options: {unknown}")
    validated = {
        **_DEFAULT_OPTIONS,
        "slider_ranges": dict(_DEFAULT_OPTIONS["slider_ranges"]),
        "slider_steps": dict(_DEFAULT_OPTIONS["slider_steps"]),
        "slider_formats": dict(_DEFAULT_OPTIONS["slider_formats"]),
        "parameter_labels": dict(_DEFAULT_OPTIONS["parameter_labels"]),
        "parameter_groups": dict(_DEFAULT_OPTIONS["parameter_groups"]),
        "parameter_order": list(_DEFAULT_OPTIONS["parameter_order"]),
        "plot_colors": dict(_DEFAULT_OPTIONS["plot_colors"]),
        "initial_visibility": dict(_DEFAULT_OPTIONS["initial_visibility"]),
    }
    for key, value in supplied.items():
        if key in {"plot_colors", "initial_visibility"}:
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping")
            extra = sorted(set(value) - set(validated[key]))
            if extra:
                raise ValueError(f"unknown {key}: {extra}")
            validated[key].update(value)
        elif key in _NAMED_OPTION_MAPS:
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping")
            extra = sorted(set(value) - set(_PARAMETER_NAMES))
            if extra:
                raise ValueError(f"unknown parameter names in {key}: {extra}")
            validated[key] = dict(value)
        elif key == "parameter_order":
            if list(value) != list(dict.fromkeys(value)) or set(value) != set(
                _PARAMETER_NAMES
            ):
                raise ValueError(
                    "parameter_order must contain each Woods-Saxon parameter once"
                )
            validated[key] = list(value)
        else:
            validated[key] = value

    validated["allow_extrapolation"] = bool(validated["allow_extrapolation"])
    for key in ("cross_section_y_range", "potential_y_range"):
        validated[key] = _finite_range(validated[key], name=key)
    for name, value in validated["slider_ranges"].items():
        validated["slider_ranges"][name] = _finite_range(
            value, name=f"slider range for {name}"
        )
    for name, value in validated["slider_steps"].items():
        step = _finite_float(value, name=f"slider step for {name}")
        if step <= 0.0:
            raise ValueError(f"slider step for {name} must be positive")
        validated["slider_steps"][name] = step
    for name, value in validated["slider_formats"].items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 12:
            raise ValueError(f"slider format for {name} must be an integer from 0 to 12")
    for key in ("title", "subtitle", "footer_note"):
        if validated[key] is not None and not isinstance(validated[key], str):
            raise ValueError(f"{key} must be text or None")
    for key, value in validated["plot_colors"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"plot color {key} must be non-empty text")
    for key, value in validated["initial_visibility"].items():
        if not isinstance(value, bool):
            raise ValueError(f"initial visibility {key} must be true or false")
    return validated


def _validate_emulator(emulator: object) -> tuple[dict[str, object], np.ndarray]:
    config = emulator.config
    if tuple(config.target) != (40, 20) or tuple(config.projectile) != (1, 0):
        raise ValueError("the first explorer supports only 40Ca+n")
    if not math.isclose(float(config.lab_energy), 14.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the first explorer requires lab_energy=14.1 MeV")
    if config.potential.name != "full_woods-saxon":
        raise ValueError("the first explorer requires full_woods-saxon")
    if tuple(emulator.parameter_names) != _PARAMETER_NAMES:
        raise ValueError("the explorer requires the ten full Woods-Saxon parameters")
    if tuple(emulator.partial_waves) != tuple(range(11)):
        raise ValueError("the first explorer requires partial waves l=0..10")
    if not math.isclose(float(emulator.kinematics.eta), 0.0, abs_tol=1e-14):
        raise ValueError("the first explorer supports neutral scattering only")
    k = _finite_float(emulator.kinematics.k, name="kinematics.k")
    if k <= 0.0:
        raise ValueError("kinematics.k must be positive")

    training_options = emulator.training_options or {}
    for key, expected in _REQUIRED_TRAINING_OPTIONS.items():
        if training_options.get(key) != expected:
            raise ValueError(f"the explorer requires {key}={expected}")
    angles = np.asarray(training_options.get("angles_degrees"), dtype=float)
    if angles.ndim != 1 or not np.array_equal(angles, np.arange(1.0, 180.0)):
        raise ValueError("the first explorer requires angles 1..179 degrees")

    design = emulator.samples.design
    training = np.asarray(design.training.values, dtype=float)
    testing = np.asarray(design.testing.values, dtype=float)
    if (
        training.ndim != 2
        or testing.ndim != 2
        or training.shape[1] != len(_PARAMETER_NAMES)
        or testing.shape[1] != len(_PARAMETER_NAMES)
        or not np.all(np.isfinite(training))
        or not np.all(np.isfinite(testing))
    ):
        raise ValueError("training and testing designs must be finite ten-column arrays")
    bounds = np.stack((training.min(axis=0), training.max(axis=0)), axis=1)

    cache = _cross_section_cache(emulator=emulator)
    missing = sorted((set(_MODEL_CACHE_KEYS) | {"sae"}) - set(cache))
    if missing:
        raise ValueError(f"packed cross-section cache is missing: {missing}")
    if tuple(cache["feature_shape"]) != (21, 6):
        raise ValueError("packed cross-section cache must have shape (21, 6)")
    if np.asarray(cache["matrices"]).shape != (21, 6, 6, 6):
        raise ValueError("packed RF matrices must have shape (21, 6, 6, 6)")
    return cache, bounds


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("complex model values must be finite")
        return [float(value.real), float(value.imag)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model values must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported HTML data value: {type(value).__name__}")


def _central_lrom_reference(
    *,
    emulator: object,
    cache: Mapping[str, object],
    values: np.ndarray,
) -> np.ndarray:
    features = _packed_effective_interaction_features(
        emulator=emulator,
        values=values,
        cache=cache,
    )
    coordinates = _solve_packed_coordinates(features=features, cache=cache)
    smatrix = _smatrix_from_packed_coordinates(
        coordinates=coordinates,
        cache=cache,
    )
    p_l = np.asarray(cache["sae"].P_l_costheta)
    p_1_l = np.asarray(cache["sae"].P_1_l_costheta)
    amplitude_a = np.zeros(p_l.shape[1], dtype=complex)
    amplitude_b = np.zeros(p_l.shape[1], dtype=complex)
    k = _rose_momentum(cache, values)
    for ell in range(11):
        amplitude_a += p_l[ell] / (2j * k) * (
            (ell + 1) * (smatrix.splus[0, ell] - 1.0)
            + ell * (smatrix.sminus[0, ell] - 1.0)
        )
        amplitude_b += p_1_l[ell] / (2j * k) * (
            smatrix.splus[0, ell] - smatrix.sminus[0, ell]
        )
    return 10.0 * (np.abs(amplitude_a) ** 2 + np.abs(amplitude_b) ** 2)


def _rose_momentum(cache: Mapping[str, object], values: np.ndarray) -> float:
    """Use the momentum normalization applied by ROSE's public cross section."""
    interaction = cache["sae"].rbes[0][0].interaction
    momentum = _finite_float(interaction.momentum(values), name="ROSE momentum")
    if momentum <= 0.0:
        raise ValueError("ROSE momentum must be positive")
    return momentum


def _fom_reference(
    value: Mapping[str, object] | None,
    *,
    angles: np.ndarray,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "angles_degrees",
        "cross_section",
    }:
        raise ValueError(
            "central_fom_evaluation requires angles_degrees and cross_section"
        )
    supplied_angles = np.asarray(value["angles_degrees"], dtype=float)
    cross_section = np.asarray(value["cross_section"], dtype=float)
    if not np.array_equal(supplied_angles, angles):
        raise ValueError("Central FOM Evaluation angles must match the emulator")
    if (
        cross_section.shape != angles.shape
        or not np.all(np.isfinite(cross_section))
        or np.any(cross_section < 0.0)
    ):
        raise ValueError("Central FOM Evaluation must be finite and non-negative")
    return {
        "angles_degrees": supplied_angles,
        "cross_section": cross_section,
    }


def _build_html_data(
    emulator: object,
    central_fom_evaluation: Mapping[str, object] | None,
    options: Mapping[str, object] | None,
) -> dict[str, object]:
    validated_options = _validated_options(options)
    cache, bounds = _validate_emulator(emulator)
    angles = np.asarray(cache["angles_degrees"], dtype=float)
    central_values = np.asarray(
        [emulator.central_parameters[name] for name in _PARAMETER_NAMES],
        dtype=float,
    )
    if not np.all(np.isfinite(central_values)):
        raise ValueError("central parameter values must be finite")

    parameter_records = []
    for name in validated_options["parameter_order"]:
        index = _PARAMETER_NAMES.index(name)
        training_min, training_max = bounds[index]
        slider_min, slider_max = validated_options["slider_ranges"].get(
            name, (training_min, training_max)
        )
        if (
            (slider_min < training_min or slider_max > training_max)
            and not validated_options["allow_extrapolation"]
        ):
            raise ValueError(
                f"slider range for {name} exceeds training bounds; "
                "set allow_extrapolation=True"
            )
        span = slider_max - slider_min
        parameter_records.append(
            {
                "name": name,
                "model_index": index,
                "label": validated_options["parameter_labels"].get(
                    name, _PARAMETER_LABELS[name]
                ),
                "group": validated_options["parameter_groups"].get(
                    name, _PARAMETER_GROUPS[name]
                ),
                "central": central_values[index],
                "training_min": training_min,
                "training_max": training_max,
                "slider_min": slider_min,
                "slider_max": slider_max,
                "step": validated_options["slider_steps"].get(
                    name, span / 1000.0
                ),
                "precision": validated_options["slider_formats"].get(name, 3),
            }
        )

    model = {key: cache[key] for key in _MODEL_CACHE_KEYS}
    model.update(
        {
            "channels": [
                {"ell": int(ell), "spin": int(spin)}
                for ell, spin in zip(cache["ell"], cache["spin"])
            ],
            "momentum_k": _rose_momentum(cache, central_values),
            "mass_pion_squared": float(MASS_PION**2),
            "p_l_costheta": cache["sae"].P_l_costheta,
            "p_1_l_costheta": cache["sae"].P_1_l_costheta,
            "potential_radii_fm": np.linspace(0.05, 10.0, 240),
        }
    )
    central_cross_section = _central_lrom_reference(
        emulator=emulator,
        cache=cache,
        values=central_values,
    )
    html_data = {
        "schema_version": HTML_SCHEMA_VERSION,
        "science": {
            "reaction_label": "40Ca+n",
            "target": tuple(emulator.config.target),
            "projectile": tuple(emulator.config.projectile),
            "lab_energy_mev": float(emulator.config.lab_energy),
            "basis_size": 6,
            "predictor_count": 6,
            "l_max": 10,
            "partial_waves": tuple(emulator.partial_waves),
            "angles_degrees": angles,
            "cross_section_units": "mb/sr",
            "radius_units": "fm",
            "model_version": __version__,
        },
        "parameters": parameter_records,
        "model": model,
        "references": {
            "central_lrom": {
                "angles_degrees": angles,
                "cross_section": central_cross_section,
            },
            "central_fom_evaluation": _fom_reference(
                central_fom_evaluation,
                angles=angles,
            ),
        },
        "options": validated_options,
    }
    return _json_value(html_data)


def _render_html(html_data: Mapping[str, object]) -> str:
    encoded = json.dumps(
        html_data,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return _HTML_TEMPLATE.replace(_HTML_DATA_MARKER, encoded, 1)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def html_explorer(
    emulator: object,
    output_path: str | Path,
    *,
    central_fom_evaluation: Mapping[str, object] | None = None,
    options: Mapping[str, object] | None = None,
) -> Path:
    """Write one trained v2 model as a self-contained HTML explorer."""
    html_data = _build_html_data(emulator, central_fom_evaluation, options)
    path = Path(output_path).expanduser().resolve()
    _write_atomic(path, _render_html(html_data))
    return path
