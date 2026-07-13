const pptxgen = require("/Users/Kitkat/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs/dist/pptxgen.cjs.js");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.subject = "Notebook 2 LROM code flow and physics risk review";
pptx.title = "Notebook 2 LROM Code Review";
pptx.company = "Learning Reduced Order Models";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.margin = 0;

const W = 13.333;
const H = 7.5;
const C = {
  bg: "F7F4EC",
  ink: "17202A",
  muted: "5F6A72",
  hair: "C9C1B3",
  teal: "1B6B6F",
  teal2: "DDEBEC",
  amber: "C8832B",
  amber2: "F2E1C7",
  red: "A6423A",
  red2: "F3D6D3",
  green: "356A43",
  green2: "DDE8DE",
  codeBg: "20252B",
  white: "FFFFFF",
};

function addBg(slide) {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: H,
    fill: { color: C.bg },
    line: { color: C.bg },
  });
}

function footer(slide, n) {
  slide.addText("Notebook 2 LROM walkthrough | code + physics-risk map", {
    x: 0.45,
    y: 7.12,
    w: 8.5,
    h: 0.18,
    fontFace: "Aptos",
    fontSize: 7.6,
    color: C.muted,
    margin: 0,
  });
  slide.addText(String(n).padStart(2, "0"), {
    x: 12.2,
    y: 7.07,
    w: 0.75,
    h: 0.24,
    fontFace: "Aptos",
    fontSize: 9,
    bold: true,
    align: "right",
    color: C.muted,
    margin: 0,
  });
}

function title(slide, kicker, claim, sub) {
  slide.addText(kicker.toUpperCase(), {
    x: 0.55,
    y: 0.35,
    w: 2.9,
    h: 0.22,
    fontFace: "Aptos",
    fontSize: 8,
    bold: true,
    color: C.teal,
    charSpace: 1.2,
    margin: 0,
  });
  slide.addText(claim, {
    x: 0.55,
    y: 0.65,
    w: 11.7,
    h: 0.65,
    fontFace: "Aptos Display",
    fontSize: 25,
    bold: true,
    color: C.ink,
    margin: 0,
    fit: "shrink",
  });
  if (sub) {
    slide.addText(sub, {
      x: 0.58,
      y: 1.27,
      w: 11.0,
      h: 0.34,
      fontFace: "Aptos",
      fontSize: 10.5,
      color: C.muted,
      margin: 0,
      fit: "shrink",
    });
  }
}

function box(slide, text, x, y, w, h, opts = {}) {
  const fill = opts.fill || "FFFFFF";
  const line = opts.line || C.hair;
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: fill, transparency: opts.transparency || 0 },
    line: { color: line, width: opts.lineWidth || 0.8 },
  });
  slide.addText(text, {
    x: x + 0.12,
    y: y + 0.1,
    w: w - 0.24,
    h: h - 0.18,
    fontFace: opts.fontFace || "Aptos",
    fontSize: opts.fontSize || 10,
    bold: opts.bold || false,
    color: opts.color || C.ink,
    margin: 0.02,
    valign: "mid",
    breakLine: false,
    fit: "shrink",
  });
}

function callout(slide, label, body, x, y, w, h, color, fill) {
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 0.1,
    y,
    w: w - 0.1,
    h,
    fill: { color: fill, transparency: 24 },
    line: { color: fill, transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.08,
    h,
    fill: { color },
    line: { color },
  });
  slide.addText(label, {
    x: x + 0.18,
    y: y + 0.08,
    w: w - 0.25,
    h: 0.2,
    fontFace: "Aptos",
    fontSize: 7.6,
    bold: true,
    color,
    margin: 0,
    charSpace: 0.5,
  });
  slide.addText(body, {
    x: x + 0.18,
    y: y + 0.32,
    w: w - 0.25,
    h: h - 0.38,
    fontFace: "Aptos",
    fontSize: 9.2,
    color: C.ink,
    margin: 0,
    fit: "shrink",
  });
}

function arrow(slide, x1, y1, x2, y2, color = C.teal, dashed = false) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      width: 1.2,
      beginArrowType: "none",
      endArrowType: "triangle",
      dash: dashed ? "dash" : "solid",
    },
  });
}

function codeBlock(slide, code, x, y, w, h, label) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.07,
    fill: { color: C.codeBg },
    line: { color: "111111", width: 0.5 },
  });
  if (label) {
    slide.addText(label, {
      x: x + 0.16,
      y: y + 0.1,
      w: w - 0.3,
      h: 0.2,
      fontFace: "Aptos",
      fontSize: 7.5,
      bold: true,
      color: "B9C4CC",
      margin: 0,
    });
  }
  slide.addText(code, {
    x: x + 0.16,
    y: y + (label ? 0.38 : 0.15),
    w: w - 0.32,
    h: h - (label ? 0.48 : 0.25),
    fontFace: "Courier New",
    fontSize: 7.1,
    color: "F2F2F2",
    margin: 0,
    breakLine: false,
    fit: "shrink",
  });
}

function bulletList(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  items.forEach((item, i) => {
    runs.push({
      text: `${i + 1}. ${item}\n`,
      options: {
        fontFace: "Aptos",
        fontSize: opts.fontSize || 10.2,
        color: opts.color || C.ink,
        breakLine: false,
      },
    });
  });
  slide.addText(runs, { x, y, w, h, margin: 0, fit: "shrink" });
}

// Slide 1
{
  const slide = pptx.addSlide();
  addBg(slide);
  slide.addText("Notebook 2 LROM Code Review", {
    x: 0.65,
    y: 0.65,
    w: 8.8,
    h: 0.8,
    fontFace: "Aptos Display",
    fontSize: 34,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText("How the code turns ROSE-backed data into LROM wavefunction diagnostics, and where the physics can be misread.", {
    x: 0.68,
    y: 1.55,
    w: 8.6,
    h: 0.6,
    fontFace: "Aptos",
    fontSize: 15,
    color: C.muted,
    margin: 0,
    fit: "shrink",
  });
  box(slide, "CODE PATH\ncore.py builds datasets\nsimple_lrom.py learns/predicts\nNotebook 2 plots diagnostics", 0.75, 3.0, 3.3, 1.75, { fill: C.teal2, line: C.teal, bold: true });
  box(slide, "PHYSICS AUDIT\nreal WS simplification\nlow partial waves\nunweighted error norm\nwide extrapolation", 4.45, 3.0, 3.3, 1.75, { fill: C.amber2, line: C.amber, bold: true });
  box(slide, "DELIVERABLE\nstudy map + risk register\nfor reviewing Notebook 2", 8.15, 3.0, 3.3, 1.75, { fill: C.red2, line: C.red, bold: true });
  slide.addText("Assumption: ROSE library calls are used correctly.", {
    x: 0.75,
    y: 6.2,
    w: 7.5,
    h: 0.28,
    fontFace: "Aptos",
    fontSize: 10,
    bold: true,
    color: C.teal,
    margin: 0,
  });
  footer(slide, 1);
}

// Slide 2
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Thesis", "Notebook 2 is a method walkthrough, not a validated scattering result.", "The code is useful for isolating LROM algebra, but the physics setting is intentionally simplified.");
  const items = [
    "core.py constructs ROSE-backed reduced bases and LS coefficient targets.",
    "simple_lrom.py fits a learned implicit reduced equation.",
    "Notebook 2 compares coefficient predictions and reconstructed wavefunctions.",
    "Physics interpretation is limited by potential simplification, low-channel scope, and error metrics."
  ];
  bulletList(slide, items, 0.75, 2.0, 5.4, 2.2, { fontSize: 13 });
  codeBlock(slide, "V(r) = -Vv * f_WS(r; Rv, av)\n\nNo imaginary absorption\nNo surface term\nNo spin-orbit term\nReal teaching case only", 7.0, 2.0, 4.55, 2.25, "Notebook 2 simplification");
  callout(slide, "READING RULE", "Treat results as controlled algebra diagnostics unless the same conclusions survive full KD optical model and observable-level tests.", 0.75, 5.15, 10.8, 0.9, C.red, C.red2);
  footer(slide, 2);
}

// Slide 3
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Pipeline", "The notebook output is a chain of data products, not a single model call.", "Each stage creates an output that becomes the input to the next stage.");
  const y = 2.0;
  const xs = [0.55, 2.65, 4.75, 6.85, 8.95, 11.05];
  const labels = [
    "core.py dataset\ncoeff_train\nphi_basis_test\nROSE basis",
    "predictors\nraw params or\npotential values",
    "fit_central_lrom\nCentralLROM\nM_j, b_j",
    "predict_coefficients\ncoeff_pred",
    "reconstruct\nphi_pred",
    "relative_l2_rows\nerror curves\nplots"
  ];
  labels.forEach((t, i) => box(slide, t, xs[i], y, 1.62, 1.25, { fill: i === 0 ? C.teal2 : i === 1 ? C.amber2 : "FFFFFF", line: i === 0 ? C.teal : i === 1 ? C.amber : C.hair, fontSize: 8.2, bold: i < 3 }));
  for (let i = 0; i < xs.length - 1; i++) arrow(slide, xs[i] + 1.62, y + 0.63, xs[i + 1] - 0.05, y + 0.63);
  codeBlock(slide, "fit = slrom.fit_central_lrom(..., p_train, data.coeff_train)\ncoeff = slrom.predict_coefficients(fit, p_test)\nphi = slrom.reconstruct_from_basis(data.rbe.basis, coeff)\nerr = slrom.relative_l2_rows(phi, data.phi_basis_test)", 1.0, 4.35, 10.8, 1.35, "Repeated Notebook 2 pattern");
  footer(slide, 3);
}

// Slide 4
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Raw-Parameter Branch", "The simplest branch asks whether parameter deltas are enough.", "This is the linear Vv/Rv path used for the broad-box stress test.");
  codeBlock(slide, "def centered_parameter_predictors(samples, center, scales):\n    return (samples - center) / scales", 0.75, 1.85, 5.0, 0.95, "simple_lrom.py");
  box(slide, "train_samples[:, :2]\ncenter[:2]\nscales[:2]", 0.95, 3.25, 2.0, 0.8, { fill: C.teal2, line: C.teal, fontSize: 8.5 });
  box(slide, "centered_parameter_predictors", 3.55, 3.25, 2.35, 0.8, { fill: C.amber2, line: C.amber, bold: true, fontSize: 8.5 });
  box(slide, "p_train / p_test\nnormalized deltas", 6.5, 3.25, 2.1, 0.8, { fill: "FFFFFF", line: C.hair, fontSize: 8.5 });
  box(slide, "fit + predict\ncoeff_linear", 9.1, 3.25, 2.0, 0.8, { fill: "FFFFFF", line: C.hair, fontSize: 8.5 });
  arrow(slide, 2.95, 3.65, 3.55, 3.65);
  arrow(slide, 5.9, 3.65, 6.5, 3.65);
  arrow(slide, 8.6, 3.65, 9.1, 3.65);
  callout(slide, "PHYSICS RISK", "A raw parameter delta knows that Rv changed, but not how the potential surface moved in radius. This is why the branch is expected to struggle over broad boxes.", 0.75, 5.2, 10.9, 0.9, C.amber, C.amber2);
  footer(slide, 4);
}

// Slide 5
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Potential-Predictor Branch", "The richer branch gives the LROM samples of the operator itself.", "Potential predictors encode shape changes at selected operator-grid points.");
  box(slide, "ROSE-derived\ninteraction.tilde()", 0.7, 2.0, 1.65, 0.8, { fill: C.teal2, line: C.teal, fontSize: 8.5, bold: true });
  box(slide, "train_alphas\ncentral_alpha\nrho_mesh", 0.7, 3.05, 1.65, 0.9, { fill: C.teal2, line: C.teal, fontSize: 8.5 });
  box(slide, "delta = U(s; alpha_i)\n- U(s; alpha_c)", 3.0, 2.45, 2.1, 0.95, { fill: "FFFFFF", line: C.hair, fontSize: 8.2 });
  box(slide, "SVD of delta\n+ greedy_maxvol", 5.75, 2.45, 2.0, 0.95, { fill: C.amber2, line: C.amber, fontSize: 8.2, bold: true });
  box(slide, "PredictorPack\ns_points\ncenter_values\nscales", 8.45, 2.25, 1.95, 1.35, { fill: "FFFFFF", line: C.hair, fontSize: 8.2 });
  box(slide, "centered_potential_predictors\np_j = [U(s_j)-U_c(s_j)] / scale_j", 10.8, 2.2, 2.0, 1.45, { fill: C.green2, line: C.green, fontSize: 7.6, bold: true });
  arrow(slide, 2.35, 2.65, 3.0, 2.9);
  arrow(slide, 5.1, 2.9, 5.75, 2.9);
  arrow(slide, 7.75, 2.9, 8.45, 2.9);
  arrow(slide, 10.4, 2.9, 10.8, 2.9);
  codeBlock(slide, "pack = delta_maxvol_predictor_pack(interaction, train_alphas, central_alpha, rho_mesh, K)\np_train = centered_potential_predictors(interaction, train_alphas, pack)\np_test  = centered_potential_predictors(interaction, test_alphas, pack)", 1.0, 5.0, 11.2, 1.1, "Notebook 2 potential-predictor section");
  footer(slide, 5);
}

// Slide 6
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Fit / Predict / Reconstruct", "Both predictor branches use the same learned implicit equation.", "The branch only changes p(alpha); the LROM training and evaluation machinery is shared.");
  codeBlock(slide, "Residual equation used in fitting:\n\na_i + sum_j p_ij M_j a_i - sum_j p_ij b_j = 0\n\nPrediction equation:\n\n[I + sum_j p_j(alpha) M_j] a = sum_j p_j(alpha) b_j", 0.75, 1.85, 5.2, 2.4, "Central-gauge LROM");
  box(slide, "fit_central_lrom\nlearns M_j and b_j\nfrom p_train and coeff_train", 6.7, 1.8, 2.4, 1.0, { fill: C.amber2, line: C.amber, bold: true, fontSize: 8.6 });
  box(slide, "predict_coefficients\nsolves a small dense system\nfor each test point", 6.7, 3.05, 2.4, 1.0, { fill: "FFFFFF", line: C.hair, fontSize: 8.6 });
  box(slide, "reconstruct_from_basis\nphi_hat = phi_0 + Phi a", 9.85, 2.42, 2.35, 1.0, { fill: C.teal2, line: C.teal, bold: true, fontSize: 8.6 });
  arrow(slide, 9.1, 2.3, 9.85, 2.9);
  arrow(slide, 9.1, 3.55, 9.85, 2.95);
  callout(slide, "INTERPRETATION", "Good coefficient prediction means the learned reduced equation matches the LS projection targets. It does not by itself validate the physical potential model.", 0.75, 5.55, 10.8, 0.82, C.teal, C.teal2);
  footer(slide, 6);
}

// Slide 7
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Notebook Outputs", "The final slides judge equation learning, basis truncation, and predictor quality.", "Notebook 2 does this through coefficient and wavefunction diagnostics.");
  const y = 1.9;
  box(slide, "Coefficient trajectories\nLS target vs LROM", 0.75, y, 2.55, 1.05, { fill: "FFFFFF", line: C.hair, bold: true });
  box(slide, "Coefficient errors\nabsolute |Delta a_j|", 3.75, y, 2.55, 1.05, { fill: "FFFFFF", line: C.hair, bold: true });
  box(slide, "Wavefunction errors\nrelative L2 rows", 6.75, y, 2.55, 1.05, { fill: "FFFFFF", line: C.hair, bold: true });
  box(slide, "Predictor visualizations\npoints + rainbows", 9.75, y, 2.55, 1.05, { fill: "FFFFFF", line: C.hair, bold: true });
  arrow(slide, 3.3, y + 0.53, 3.75, y + 0.53);
  arrow(slide, 6.3, y + 0.53, 6.75, y + 0.53);
  arrow(slide, 9.3, y + 0.53, 9.75, y + 0.53);
  codeBlock(slide, "phi_pred = reconstruct_from_basis(data.rbe.basis, coeff_pred)\nwf_error = relative_l2_rows(phi_pred, data.phi_basis_test)\n\nNotebook output: plots of coefficient error and wavefunction error.", 1.0, 4.0, 11.0, 1.25, "Final diagnostic calculation");
  callout(slide, "IMPORTANT", "The LS floor separates basis truncation from equation-learning error; compare LROM to its own LS floor before comparing to ROSE ROM.", 1.0, 5.85, 10.7, 0.62, C.red, C.red2);
  footer(slide, 7);
}

// Slide 8
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Physics-Risk Register", "The largest risks are interpretation errors, not obvious code crashes.", "These are places where correct code can still support an overbroad physics claim.");
  const risks = [
    ["Simplified interaction", "real WS only; no absorption, surface, or spin-orbit terms"],
    ["Low channel scope", "examples use l_max=1; not a full observable calculation"],
    ["Unweighted plotted norm", "LS projection is quadrature-weighted; displayed relative L2 is plain row norm"],
    ["Wide extrapolation", "test scans extend far beyond training ranges"],
    ["Basis mismatch", "ROSE baseline and central LROM can live in different reduced coordinates"],
    ["Real-part plots", "coefficient trajectory plots can hide imaginary behavior in complex cases"]
  ];
  risks.forEach((r, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.75 + col * 6.05;
    const y = 1.75 + row * 1.45;
    box(slide, `${r[0]}\n${r[1]}`, x, y, 5.3, 1.05, { fill: i < 2 ? C.red2 : i < 4 ? C.amber2 : "FFFFFF", line: i < 2 ? C.red : i < 4 ? C.amber : C.hair, fontSize: 9.2, bold: true });
  });
  footer(slide, 8);
}

// Slide 9
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Verification Plan", "Trust the notebook by testing the assumptions one at a time.", "The goal is to distinguish code flow, numerical fit quality, and physical validity.");
  const items = [
    "Add weighted wavefunction error using the same trapezoid weights as LS projection.",
    "Plot imaginary coefficient components when the full complex KD model is used.",
    "Record condition numbers of learned implicit matrices across test scans.",
    "Repeat a smaller section with the full KD optical model, not only real WS.",
    "Run n_phi and n_U sweeps to separate basis error from LROM equation error.",
    "Add observable-level checks before claiming scattering accuracy."
  ];
  bulletList(slide, items, 0.8, 1.75, 6.25, 4.5, { fontSize: 12.2 });
  box(slide, "Suggested first test\n\nReplace relative_l2_rows with a weighted relative L2 helper, then compare whether the ranking of ROSE, linear LROM, and predictor LROM changes.", 7.65, 2.0, 4.35, 2.2, { fill: C.green2, line: C.green, fontSize: 11, bold: true });
  box(slide, "Suggested second test\n\nKeep the same code flow but use the full optical model branch. If conclusions change, the teaching example was too special.", 7.65, 4.55, 4.35, 1.65, { fill: C.amber2, line: C.amber, fontSize: 10.5, bold: true });
  footer(slide, 9);
}

// Slide 10
{
  const slide = pptx.addSlide();
  addBg(slide);
  title(slide, "Study Path", "Read notebook 2 from outputs backward.", "Start with each plot, then trace the exact data product that created it.");
  const steps = [
    ["1", "Final plots", "wavefunction errors, coefficient trajectories, predictor rainbows"],
    ["2", "Error arrays", "relative_l2_rows(phi_pred, phi_basis_test)"],
    ["3", "Predicted phi", "reconstruct_from_basis(ROSE basis, coeff_pred)"],
    ["4", "Predicted coeffs", "predict_coefficients(CentralLROM, p_test)"],
    ["5", "Fit object", "fit_central_lrom(p_train, coeff_train)"],
    ["6", "Predictors", "raw deltas or potential values from selected s_points"]
  ];
  steps.forEach((s, i) => {
    const x = 0.75 + (i % 3) * 4.05;
    const y = 1.85 + Math.floor(i / 3) * 1.8;
    slide.addShape(pptx.ShapeType.ellipse, { x, y: y + 0.05, w: 0.36, h: 0.36, fill: { color: C.teal }, line: { color: C.teal } });
    slide.addText(s[0], { x, y: y + 0.12, w: 0.36, h: 0.14, fontFace: "Aptos", fontSize: 8, bold: true, color: C.white, align: "center", margin: 0 });
    box(slide, `${s[1]}\n${s[2]}`, x + 0.45, y, 3.1, 0.72, { fill: "FFFFFF", line: C.hair, fontSize: 8.4, bold: true });
  });
  callout(slide, "BOTTOM LINE", "Notebook 2 is strongest when used as a transparent LROM algebra lab. Its physics claims become stronger only after weighted errors, full complex potentials, larger partial-wave coverage, and observable checks.", 0.85, 6.0, 11.3, 0.8, C.teal, C.teal2);
  footer(slide, 10);
}

(async () => {
  await pptx.writeFile({
    fileName: "/Users/Kitkat/.codex/worktrees/aee7/Learning-Reduced-Order-Models/outputs/manual-20260601-lrom/presentations/notebook2-lrom-code-review/output/notebook2-lrom-code-review.pptx",
  });
})();
