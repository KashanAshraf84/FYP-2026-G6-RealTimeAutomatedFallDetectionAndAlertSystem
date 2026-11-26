// Parity test: JavaScript port (pipeline.js + ONNX) vs Python (not deployed).
// node test_parity.mjs <fixture dir>   (needs onnxruntime-node)
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { PersonTracker, detectorStep, decodePoses, letterbox } from './pipeline.js';

const require = createRequire(process.cwd() + '/');
const ort = require('onnxruntime-node');
const dir = process.argv[2];
const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, '$1'));

const fall = await ort.InferenceSession.create(path.join(here, 'models', 'fallnet_v2.onnx'));
const runModel = async (batch, n) =>
  (await fall.run({ features: new ort.Tensor('float32', batch, [n, 30, 62]) })).probs.data;

let frames = 0, people = 0, idMismatch = 0, maxDiff = 0, labelMismatch = 0;
for (const replay of JSON.parse(fs.readFileSync(path.join(dir, 'replays.json')))) {
  const tracker = new PersonTracker();
  for (const f of replay.frames) {
    const preds = await detectorStep(tracker, f.t, { kps: f.kps, boxes: f.boxes }, replay.H, runModel);
    frames++;
    const got = preds.map(p => p.trackId).join(','), want = f.expected.map(e => e.id).join(',');
    if (got !== want) { idMismatch++; if (idMismatch < 4) console.log('ID mismatch', replay.clip, f.t, got, '!=', want); continue; }
    f.expected.forEach((e, i) => {
      people++;
      if ((e.probs === null) !== (preds[i].probs === null)) { labelMismatch++; return; }
      if (!e.probs) return;
      maxDiff = Math.max(maxDiff, ...e.probs.map((v, j) => Math.abs(v - preds[i].probs[j])));
      if (e.probs.indexOf(Math.max(...e.probs)) !== preds[i].probs.indexOf(Math.max(...preds[i].probs))) labelMismatch++;
    });
  }
}
console.log(`tracker+model: ${frames} frames, ${people} person-predictions, ID mismatches ${idMismatch}, ` +
  `label mismatches ${labelMismatch}, max |prob diff| ${maxDiff.toExponential(2)}`);

const yolo = await ort.InferenceSession.create(path.join(here, 'models', 'yolov8n-pose-640.onnx'));
for (const f of JSON.parse(fs.readFileSync(path.join(dir, 'frames.json')))) {
  const buf = fs.readFileSync(path.join(dir, f.file));
  const tensor = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const out = await yolo.run({ [yolo.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, 640, 640]) });
  const raw = out[yolo.outputNames[0]];
  const det = decodePoses(raw.data, raw.dims[2], letterbox(f.w, f.h));
  const kpDiff = det.kps.length && f.kps.length
    ? Math.max(...det.kps[0].map((p, j) => Math.hypot(p[0] - f.kps[0][j][0], p[1] - f.kps[0][j][1]))) : NaN;
  console.log(`YOLO ${f.file}: persons JS ${det.boxes.length} vs Ultralytics ${f.boxes.length}; ` +
    `top conf ${det.boxes[0]?.[4].toFixed(3)} vs ${f.boxes[0]?.[4].toFixed(3)}; max keypoint offset ${kpDiff.toFixed(2)} px`);
}
