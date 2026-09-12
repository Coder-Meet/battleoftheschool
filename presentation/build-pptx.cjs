const fs = require("node:fs");
const path = require("node:path");
const pptxgen = require("pptxgenjs");

function containPng(filename, box) {
  const bytes = fs.readFileSync(filename);
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (bytes.length < 24 || !bytes.subarray(0, 8).equals(signature)) {
    throw new Error(`Only generated PNG assets are accepted: ${filename}`);
  }
  const width = bytes.readUInt32BE(16);
  const height = bytes.readUInt32BE(20);
  if (!width || !height || width > 10000 || height > 10000) {
    throw new Error(`Invalid PNG dimensions: ${filename}`);
  }
  const scale = Math.min(box.w / width, box.h / height);
  const w = width * scale,
    h = height * scale;
  return { x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h };
}

const root = path.resolve(process.argv[2] || "outputs/presentation-kit");
const slides = JSON.parse(
  fs.readFileSync(path.join(root, "deck.json"), "utf8"),
);
const pptx = new pptxgen();
pptx.defineLayout({ name: "BRANCHSEED", width: 13.333333, height: 7.5 });
pptx.layout = "BRANCHSEED";
pptx.author = "Branchseed team";
pptx.subject = "Toralis Labs challenge — five-minute demonstration";
pptx.title = "Find the branch. Keep the evidence.";
pptx.company = "Branchseed";
pptx.lang = "en-CA";
pptx.theme = {
  headFontFace: "Branchseed Display",
  bodyFontFace: "Branchseed Text",
  lang: "en-CA",
};
slides.forEach((model, index) => {
  const slide = pptx.addSlide();
  slide.background = { color: model.background };
  model.elements.forEach((e) => {
    const box = { x: e.x / 144, y: e.y / 144, w: e.w / 144, h: e.h / 144 };
    switch (e.kind) {
      case "text":
        slide.addText(e.text, {
          ...box,
          fontFace:
            e.font === "display" ? "Branchseed Display" : "Branchseed Text",
          fontSize: e.size / 2,
          color: e.color,
          margin: 0,
          breakLine: false,
          valign: "top",
          paraSpaceAfterPt: 0,
          lineSpacingMultiple: 1.05,
        });
        break;
      case "image":
        slide.addImage({
          path: path.join(root, e.source),
          ...containPng(path.join(root, e.source), box),
        });
        break;
      case "line":
        slide.addShape(pptx.ShapeType.line, {
          x: Math.min(e.x, e.x + e.w) / 144,
          y: Math.min(e.y, e.y + e.h) / 144,
          w: Math.abs(e.w) / 144,
          h: Math.abs(e.h) / 144,
          flipV: e.w * e.h < 0,
          line: { color: e.color, width: e.stroke / 2 },
        });
        break;
      case "rect":
      case "circle":
        slide.addShape(
          e.kind === "rect" ? pptx.ShapeType.rect : pptx.ShapeType.ellipse,
          {
            ...box,
            line: {
              color: e.color,
              width: e.stroke / 2,
              transparency: e.stroke ? 0 : 100,
            },
            fill: { color: e.color, transparency: e.stroke ? 100 : 0 },
          },
        );
        break;
      default:
        throw new Error(`Unknown element ${e.kind}`);
    }
  });
  if (index === 4 && fs.existsSync(path.join(root, "branchseed-film.mp4"))) {
    slide.addMedia({
      type: "video",
      path: path.join(root, "branchseed-film.mp4"),
      x: (1920 - (596 * 16) / 9) / 288,
      y: 357 / 144,
      w: (596 * 16) / 9 / 144,
      h: 596 / 144,
    });
  }
  slide.addNotes(
    `Speaker ${model.speaker} — ${model.seconds} seconds.\n\n${model.notes}`,
  );
});
pptx.writeFile({ fileName: path.join(root, "branchseed-editable.pptx") });
