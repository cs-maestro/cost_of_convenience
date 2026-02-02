#!/usr/bin/env node

const fs             = require('fs');
const fsp            = fs.promises;
const path           = require('path');
const cliProgress    = require('cli-progress');
const { chromium }   = require('playwright');
const { PlaywrightBlocker } = require('@cliqz/adblocker-playwright');
const fetch          = require('cross-fetch');

const SCRIPT_DIR     = __dirname;
const MAPPING_CSV    = path.join(SCRIPT_DIR, '../../../urls_raw_data/ss_data/mapping.csv');
const SCREENSHOT_DIR = path.join(SCRIPT_DIR, '../../ss_dedupe/unique_ss');
const OUTPUT_DIR     = path.join(SCRIPT_DIR, 'har_html_data');
const OUTPUT_STATUS  = path.join(OUTPUT_DIR, 'har_html_status.csv');

const TIMEOUT_MS     = 25_000; // a little higher, HTML + HAR flush
const HEADLESS       = false;

function csvParseMappedWithScreenshot(csvText) {
  const lines = csvText.split(/\r?\n/).filter(Boolean);
  const header = lines.shift();
  // Expecting: orig_url,final_url,hash,status (quoted)
  const rowRe = /^"((?:[^"]|"")*)","((?:[^"]|"")*)","((?:[^"]|"")*)","((?:[^"]|"")*)"$/;

  const rows = [];
  for (const line of lines) {
    const m = line.match(rowRe);
    if (!m) continue;
    const [ , orig_url_q, final_url_q, hash_q, status_q ] = m;
    const unq = s => s.replace(/""/g, '"');
    const orig_url = unq(orig_url_q);
    const final_url = unq(final_url_q);
    const hash = unq(hash_q);
    const status = unq(status_q);

    if (status !== 'mapped') continue;
    const ssPath = path.join(SCREENSHOT_DIR, `${hash}.png`);
    if (!fs.existsSync(ssPath)) continue;

    rows.push({ orig_url, final_url, hash, status });
  }
  return rows;
}

async function captureForRow(row, browser, blocker, csvStream) {
  const { orig_url, hash } = row;
  const baseOut = path.join(OUTPUT_DIR, hash);

  // Skip if we already have both HAR and HTMLs
  const already =
    fs.existsSync(`${baseOut}.har`) &&
    fs.existsSync(`${baseOut}.server.html`) &&
    fs.existsSync(`${baseOut}.dom.html`);
  if (already) {
    csvStream.write(`"${orig_url}","${hash}","skipped-existing"\n`);
    return;
  }

  const context = await browser.newContext({
    // Full HAR with bodies inlined
    recordHar: {
      path: `${baseOut}.har`,
      content: 'embed', // inline content in HAR
      mode: 'full'      // keep full metadata
    },
    serviceWorkers: 'block'
  });
  const page = await context.newPage();

  page.on('dialog', async d => { try { await d.dismiss(); } catch {} });

  let status = 'ok';
  try {
    try {
      await blocker.enableBlockingInPage(page);
    } catch (e) {
      // non-fatal
    }

    const resp = await page.goto(orig_url, { waitUntil: 'networkidle', timeout: TIMEOUT_MS });

    // Save server HTML (raw response) if we have a navigation response
    if (resp) {
      try {
        const serverHtml = await resp.text();
        await fsp.writeFile(`${baseOut}.server.html`, serverHtml, 'utf8');
      } catch (e) {
        status = `partial-no-serverhtml:${e.message}`;
      }
    } else {
      status = 'partial-no-nav-response';
    }

    // Save DOM HTML after hydration
    try {
      const domHtml = await page.content();
      await fsp.writeFile(`${baseOut}.dom.html`, domHtml, 'utf8');
    } catch (e) {
      status = status.startsWith('ok') ? `partial-no-domhtml:${e.message}` : status;
    }

  } catch (e) {
    status = `error:${e.message}`;
  } finally {
    // Closing the context flushes the HAR to disk
    try { await page.close(); } catch {}
    try { await context.close(); } catch (e) {
      status = status.startsWith('ok') ? `partial-har-close:${e.message}` : status;
    }
    csvStream.write(`"${orig_url}","${hash}","${status.replace(/"/g,'""')}"\n`);
  }
}

(async () => {
  await fsp.mkdir(OUTPUT_DIR, { recursive: true });

  // Read & filter the mapping CSV
  if (!fs.existsSync(MAPPING_CSV)) {
    console.error('Mapping CSV not found at:', MAPPING_CSV);
    process.exit(1);
  }
  const mappingCsv = await fsp.readFile(MAPPING_CSV, 'utf8');
  const rows = csvParseMappedWithScreenshot(mappingCsv);

  console.log(`Eligible rows (mapped + screenshot present): ${rows.length}`);
  if (rows.length === 0) {
    console.log('Nothing to do.');
    process.exit(0);
  }

  const csvStream = fs.createWriteStream(OUTPUT_STATUS, { flags: 'a' });
  if (fs.statSync(OUTPUT_STATUS, { throwIfNoEntry: false })?.size === 0) {
    csvStream.write('orig_url,hash,status\n');
  }

  const browser = await chromium.launch({ headless: HEADLESS });
  const blocker = await PlaywrightBlocker.fromPrebuiltAdsAndTracking(fetch);

  const bar = new cliProgress.SingleBar({
    format: 'Progress |{bar}| {percentage}% || {value}/{total}',
    hideCursor: true
  }, cliProgress.Presets.shades_classic);
  bar.start(rows.length, 0);

  // Optional shuffle
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }

  for (const row of rows) {
    await captureForRow(row, browser, blocker, csvStream);
    bar.increment();
    await new Promise(r => setTimeout(r, 2000)); // small pacing
  }

  bar.stop();
  await browser.close();
  csvStream.end();
  console.log('Done. HAR + HTMLs in', OUTPUT_DIR);
})();
