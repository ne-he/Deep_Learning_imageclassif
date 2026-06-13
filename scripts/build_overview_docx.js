/* Generates "NemVision Project Overview.docx" at the repo root.
   Run: NODE_PATH=$(npm root -g) node scripts/build_overview_docx.js */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, Footer, PageNumber,
} = require("docx");

const GREEN = "1F8A2E";
const INK = "0A0A0A";
const GREY = "5A5A5A";
const LIGHT = "8A8A8A";
const CONTENT_W = 9360;

// ---------- helpers ----------
const accentRule = () =>
  new Paragraph({
    spacing: { before: 60, after: 220 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: GREEN, space: 1 } },
    children: [],
  });

const body = (runs, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 140, line: 276 },
    children: (Array.isArray(runs) ? runs : [new TextRun(runs)]),
    ...opts.p,
  });

const bullet = (text, bold) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 268 },
    children: bold
      ? [new TextRun({ text: bold, bold: true }), new TextRun({ text: text })]
      : [new TextRun(text)],
  });

const step = (n, title, text) =>
  new Paragraph({
    spacing: { after: 110, line: 272 },
    indent: { left: 360 },
    children: [
      new TextRun({ text: `${n}  `, bold: true, color: GREEN }),
      new TextRun({ text: `${title}. `, bold: true, color: INK }),
      new TextRun({ text, color: GREY }),
    ],
  });

const mono = (text) =>
  new Paragraph({
    spacing: { after: 40, line: 252 },
    shading: { fill: "F4F4F2", type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Consolas", size: 19, color: "333333" })],
  });

// ---------- results table ----------
const border = { style: BorderStyle.SINGLE, size: 1, color: "DDDDDD" };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (text, { w, head, bold, fill, right } = {}) =>
  new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 70, bottom: 70, left: 120, right: 120 },
    children: [
      new Paragraph({
        alignment: right ? AlignmentType.RIGHT : AlignmentType.LEFT,
        children: [
          new TextRun({
            text,
            bold: head || bold,
            size: head ? 18 : 20,
            color: head ? LIGHT : INK,
          }),
        ],
      }),
    ],
  });

const W = [3360, 2000, 2000, 2000];
const resRow = (cells, { head, best } = {}) =>
  new TableRow({
    children: cells.map((t, i) =>
      cell(t, {
        w: W[i],
        head,
        right: i > 0,
        bold: best,
        fill: head ? "F0F0EE" : best ? "EAF7EA" : undefined,
      })
    ),
  });

const resultsTable = new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: W,
  rows: [
    resRow(["Model", "Accuracy", "F1", "AUC"], { head: true }),
    resRow(["ResNet50  (live)", "91.6%", "0.916", "0.99"], { best: true }),
    resRow(["Ensemble (soft vote)", "91.0%", "0.911", "0.99"]),
    resRow(["MobileNetV2", "84.2%", "0.847", "0.97"]),
    resRow(["EfficientNet-B0", "80.5%", "0.806", "0.97"]),
  ],
});

// ---------- document ----------
const doc = new Document({
  creator: "Nehemiah",
  title: "NemVision Project Overview",
  styles: {
    default: { document: { run: { font: "Arial", size: 21, color: "222222" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: INK },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: INK },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 240 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "NemVision  ", size: 16, color: LIGHT, bold: true }),
            new TextRun({ text: "Waste Image Classification with Explainable AI   |   ", size: 16, color: LIGHT }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, color: LIGHT }),
          ],
        })],
      }),
    },
    children: [
      // ---- title block ----
      new Paragraph({
        spacing: { after: 20 },
        children: [new TextRun({ text: "NemVision", bold: true, size: 56, color: INK })],
      }),
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({ text: "Waste Image Classification with Explainable AI", size: 26, color: GREEN })],
      }),
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({
          text: "A deep-learning classifier that names the material of a waste item and shows the exact pixels it looked at.",
          italics: true, color: GREY, size: 21,
        })],
      }),
      accentRule(),

      // ---- overview ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("What is NemVision")] }),
      body("NemVision sorts a photo of trash into one of six material classes and, crucially, explains its answer. Instead of returning a single label from a black box, it overlays a Grad-CAM heatmap so anyone can see which part of the image drove the decision. The result is a portfolio-grade computer-vision project that is both accurate and transparent."),
      body("It is trained on TrashNet, a public dataset of 2,527 images across cardboard, glass, metal, paper, plastic, and trash. Three pretrained CNN backbones were benchmarked under one identical, seed-locked split, and the strongest of them now powers a live web demo."),

      // ---- key features ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Key Features")] }),
      bullet("classifies a waste photo into 6 material classes with a confidence score.", "Six-class prediction "),
      bullet("a Grad-CAM heatmap highlights the regions the model used, so predictions are auditable.", "Visual explanation "),
      bullet("shows the model's probability across all six classes, not just the winner.", "Full distribution "),
      bullet("three CNN architectures compared fairly, plus a soft-voting ensemble.", "Model benchmark "),
      bullet("upload a file, drag-and-drop, or try bundled samples with one click.", "Interactive demo "),
      bullet("a real PyTorch model runs server-side and returns predictions over an API.", "Live inference "),

      // ---- how it works ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("How It Works")] }),
      body("Four steps run every time an image is classified:"),
      step(1, "Preprocess", "the photo is resized to 224x224 and normalized to ImageNet statistics, the same transform the model saw during training."),
      step(2, "Classify", "a ResNet50 fine-tuned on TrashNet runs a forward pass, and a softmax turns its outputs into probabilities across the six classes."),
      step(3, "Explain", "Grad-CAM traces the gradients of the winning class back to the last convolutional layer to build a heatmap of the decisive regions."),
      step(4, "Report", "the predicted material, its confidence, the full distribution, and the heatmap overlay are returned to the user."),

      // ---- models / results ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("The Models and Results")] }),
      body("Each backbone was trained and evaluated on the same split so the comparison is honest. ResNet50 came out strongest and serves the live demo."),
      resultsTable,
      body([
        new TextRun({ text: "The hardest class is ", color: GREY }),
        new TextRun({ text: "trash", bold: true, color: INK }),
        new TextRun({ text: ", with only 137 images in the dataset. A weighted sampler and label smoothing lift its recall well above what a naive model reaches.", color: GREY }),
      ], { p: { spacing: { before: 140, after: 140, line: 276 } } }),

      // ---- design choices ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Why These Design Choices")] }),
      bullet("only the deepest block and a fresh classifier head are unfrozen, so general low-level features stay intact, which suits a small dataset.", "Selective fine-tuning "),
      bullet("inputs match the backbones' pretraining distribution, keeping the transferred features valid.", "ImageNet normalization "),
      bullet("oversample the rare trash class per batch and soften the targets, which lifts minority recall.", "Weighted sampler and label smoothing "),
      bullet("a 70/15/15 split with a test set never touched during training or tuning gives an honest final number.", "Held-out test set "),

      // ---- architecture ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Architecture and Deployment")] }),
      body("NemVision is split into a static frontend and a model-serving backend that talk over HTTP:"),
      mono("[ Vercel ]  static web UI  ->  POST /predict (image)"),
      mono("        ->  [ Hugging Face Space ]  FastAPI + ResNet50"),
      mono("        ->  JSON { label, confidence, distribution, gradcam }  ->  UI"),
      body([
        new TextRun({ text: "Frontend: ", bold: true, color: INK }),
        new TextRun({ text: "a single self-contained HTML page (hosted on Vercel) with an animated hero, the classifier widget, and content panels.", color: GREY }),
      ]),
      body([
        new TextRun({ text: "Backend: ", bold: true, color: INK }),
        new TextRun({ text: "a FastAPI service (hosted on a Hugging Face Space via Docker) that loads the trained checkpoint once and runs inference plus Grad-CAM, reusing the exact model and transform code from the training pipeline.", color: GREY }),
      ]),

      // ---- tech stack ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Tech Stack")] }),
      bullet("PyTorch and torchvision (transfer learning, fine-tuning, ensemble).", "Modeling: "),
      bullet("hook-based Grad-CAM with removable hooks.", "Explainability: "),
      bullet("FastAPI plus Uvicorn, containerized with Docker.", "Serving: "),
      bullet("HTML, CSS, and vanilla JavaScript, no framework.", "Frontend: "),
      bullet("Hugging Face Spaces for the model API, Vercel for the web UI.", "Hosting: "),
      bullet("config-driven YAML, seed-locked splits, headless reporting.", "Engineering: "),

      // ---- structure ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Project Structure")] }),
      mono("configs/    YAML configs (base + per-model, with inheritance)"),
      mono("src/        config, data, models, engine, metrics, ensemble, gradcam"),
      mono("scripts/    train.py, compare.py, gradcam.py"),
      mono("backend/    FastAPI app, Dockerfile, requirements (inference API)"),
      mono("frontend/   static web UI (index.html + assets + samples)"),
      mono("outputs/    checkpoints + metrics (gitignored)"),

      // ---- running ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Running It Yourself")] }),
      body("Train and compare the models, then generate explanations:"),
      mono("python scripts/train.py --config configs/resnet50.yaml"),
      mono("python scripts/compare.py --configs configs/resnet50.yaml \\"),
      mono("    configs/efficientnet_b0.yaml configs/mobilenet_v2.yaml"),
      mono("python scripts/gradcam.py --config configs/resnet50.yaml"),
      body("Serve the model locally:", { p: { spacing: { before: 120, after: 80 } } }),
      mono("uvicorn backend.app:app --port 7860     # then open /health"),

      // ---- links ----
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Links")] }),
      bullet("github.com/ne-he/Deep_Learning_imageclassif", "Source code: "),
      bullet("a Hugging Face Space running the ResNet50 API.", "Live model: "),
      bullet("the NemVision demo deployed on Vercel.", "Live demo: "),

      new Paragraph({
        spacing: { before: 280 },
        border: { top: { style: BorderStyle.SINGLE, size: 6, color: "DDDDDD", space: 6 } },
        children: [new TextRun({ text: "Built by Nehemiah", size: 18, color: LIGHT, italics: true })],
      }),
    ],
  }],
});

const out = path.join(__dirname, "..", "NemVision Project Overview.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("WROTE", out, buf.length, "bytes");
});
