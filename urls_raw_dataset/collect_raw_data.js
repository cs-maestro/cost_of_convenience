#!/usr/bin/env node
const fs             = require('fs');
const fsp            = fs.promises;
const path           = require('path');
const crypto         = require('crypto');
const cliProgress    = require('cli-progress');
const { chromium }   = require('playwright');
const { PlaywrightBlocker } = require('@cliqz/adblocker-playwright');
const fetch          = require('cross-fetch');

const SCRIPT_DIR  = __dirname;
const ORIG_URLS   = path.join(SCRIPT_DIR, '..', 'orig_datasets', 'unique_urls.txt');
const OUTPUT_DIR  = path.join(SCRIPT_DIR, 'ss_data');
const OUTPUT_CSV  = path.join(OUTPUT_DIR, 'mapping.csv');

const TIMEOUT_MS  = 20_000;  // ms

async function fetchAndRecord(originalUrl, browser, blocker, processedHashes, csvStream) {
  // Use originalUrl hash for consistent filenames even if navigation fails
  let tmpHash = crypto.createHash('md5').update(originalUrl).digest('hex');
  let hash    = tmpHash;
  let finalUrl = originalUrl;
  let status;

  // Create a clean context
  const context = await browser.newContext({
    serviceWorkers: 'block' // keep things deterministic
  });
  const page = await context.newPage();

  page.on('dialog', async dialog => {
    try { await dialog.dismiss(); } catch {}
  });

  try {
    try {
      await blocker.enableBlockingInPage(page);
    } catch (e) {
      console.warn(`AdBlocker error for ${originalUrl}: ${e.message}`);
    }

    const resp = await page.goto(originalUrl, {
      waitUntil: 'networkidle', // stable before capture
      timeout: TIMEOUT_MS
    });

    if (resp?.url()) finalUrl = resp.url().split('?')[0];
    hash = crypto.createHash('md5').update(finalUrl).digest('hex');

    if (processedHashes.has(hash)) {
      status = 'skipped';
    } else {
      processedHashes.add(hash);
      // Save screenshot (full page)
      await page.screenshot({
        path: path.join(OUTPUT_DIR, `${hash}.png`),
        fullPage: true
      });
      status = 'mapped';
    }

  } catch {
    finalUrl = originalUrl;
    hash     = tmpHash;
    status   = 'inaccessible';
  } finally {
    // Write to CSV
    const row = [originalUrl, finalUrl, hash, status]
      .map(s => `"${s.replace(/"/g, '""')}"`)
      .join(',');
    csvStream.write(row + '\n');

    await page.close();
    await context.close(); // close per-URL context
  }
}

;(async () => {
  const raw = (await fsp.readFile(ORIG_URLS, 'utf-8'))
                .split(/\r?\n/).filter(Boolean);
  const unique = Array.from(new Set(raw));
  console.log(`Total unique URLs in master list: ${unique.length}`);

  const seen = new Set();
  if (fs.existsSync(OUTPUT_CSV)) {
    const csv = await fsp.readFile(OUTPUT_CSV, 'utf8');
    csv.split(/\r?\n/).slice(1).forEach(line => {
      if (!line) return;
      const m = line.match(/^"([^"]*)"/);
      if (m) seen.add(m[1]);
    });
    console.log(`Skipping ${seen.size} URLs already in ${path.basename(OUTPUT_CSV)}`);
  }

  const toProcess = unique.filter(u => !seen.has(u));
  console.log(`Will process ${toProcess.length} URLs this run.`);

  // simple shuffle
  for (let i = toProcess.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [toProcess[i], toProcess[j]] = [toProcess[j], toProcess[i]];
  }

  await fsp.mkdir(OUTPUT_DIR, { recursive: true });

  const csvStream = fs.createWriteStream(OUTPUT_CSV, { flags: 'a' });
  if (seen.size === 0) {
    csvStream.write('orig_url,final_url,hash,status\n');
  }

  const browser = await chromium.launch({ headless: false });
  const blocker = await PlaywrightBlocker.fromPrebuiltAdsAndTracking(fetch);

  const bar = new cliProgress.SingleBar({
    format: 'Progress |{bar}| {percentage}% || {value}/{total} URLs',
    hideCursor: true
  }, cliProgress.Presets.shades_classic);
  bar.start(toProcess.length, 0);

  const processed = new Set(seen);
  const sleep = ms => new Promise(res => setTimeout(res, ms));

  for (const url of toProcess) {
    await fetchAndRecord(url, browser, blocker, processed, csvStream);
    bar.increment();
    await sleep(2000); // small pacing
  }

  bar.stop();
  await browser.close();
  csvStream.end();

  console.log('Done. Screenshots in', OUTPUT_DIR);
})();
