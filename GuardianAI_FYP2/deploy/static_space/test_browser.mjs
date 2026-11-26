// Automated browser test (not deployed): opens the demo in headless Edge, plays
// videos through ?video=..., waits for the end and reports alerts + a screenshot.
// node test_browser.mjs <base url> <out dir> <video1> [video2 ...]   (needs playwright-core)
import { createRequire } from 'node:module';
const require = createRequire(process.cwd() + '/');
const { chromium } = require('playwright-core');

const [base, outDir, ...videos] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true,
  args: ['--autoplay-policy=no-user-gesture-required'],
});
for (const v of videos) {
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`${base}/index.html?video=${encodeURIComponent(v)}`);
  let shot = false;
  const start = Date.now();
  while (Date.now() - start < 180000) {
    const s = await page.evaluate(() => ({ done: window.__ga.done, frames: window.__ga.frames, last: window.__ga.last }));
    if (!shot && s.last && s.last.some(p => p.label === 'fall')) { await page.screenshot({ path: `${outDir}/shot_${v}.png` }); shot = true; }
    if (s.done) break;
    await page.waitForTimeout(250);
  }
  if (!shot) await page.screenshot({ path: `${outDir}/shot_${v}.png` });
  const r = await page.evaluate(() => ({ frames: window.__ga.frames, events: window.__ga.events, timeline: window.__ga.timeline,
    backend: document.getElementById('backend-status').textContent, fps: document.getElementById('fps-status').textContent,
    error: document.getElementById('error-bar').textContent }));
  console.log(`${v}: analysed ${r.frames} frames on ${r.backend} at ${r.fps}; timeline ${r.timeline}; events:`,
    r.events.map(e => `${e.t?.toFixed(1)}s ${e.text}`).join(' | ') || 'none', r.error ? `ERROR ${r.error}` : '', errors.length ? `JS errors: ${errors.slice(0, 3)}` : '');
  await page.close();
}
await browser.close();
