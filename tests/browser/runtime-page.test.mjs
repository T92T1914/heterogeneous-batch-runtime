import assert from 'node:assert/strict';
import {after, before, test} from 'node:test';
import {createServer} from 'node:http';
import {readFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium, firefox, webkit} from 'playwright';

const root = fileURLToPath(new URL('../../web/', import.meta.url));
const data = JSON.parse(await readFile(path.join(root, 'evidence.json')));
const key = 'heterogeneous-batch-runtime.appearance.v1';
const engines = {chromium, webkit, firefox};
const mime = {'.html':'text/html', '.css':'text/css', '.js':'text/javascript', '.mjs':'text/javascript', '.json':'application/json'};
let browser, server, base;

before(async () => {
  server = createServer(async (request, response) => {
    const name = new URL(request.url, 'http://localhost').pathname.slice(1) || 'index.html';
    if (name === 'favicon.ico') { response.writeHead(204).end(); return; }
    if (!/^[a-zA-Z0-9.-]+$/.test(name)) { response.writeHead(404).end(); return; }
    try {
      const bytes = await readFile(path.join(root, name));
      response.writeHead(200, {'Content-Type':mime[path.extname(name)] ?? 'text/plain', 'Cache-Control':'no-store'}).end(bytes);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
  const engine = process.env.RUNTIME_BROWSER_ENGINE ?? 'chromium';
  if (!Object.hasOwn(engines, engine)) throw new Error('Unsupported browser engine');
  const channel = process.env.RUNTIME_BROWSER_CHANNEL;
  if (channel && (engine !== 'chromium' || !['chrome', 'chromium'].includes(channel))) throw new Error('Unsupported browser channel');
  browser = await engines[engine].launch({headless:true, ...(engine === 'chromium' ? {chromiumSandbox:true, args:['--mute-audio','--disable-gpu'], ...(channel ? {channel} : {})} : {})});
  console.log(`Owned loopback, isolated headless ${engine} ${browser.version()}, fresh contexts`);
});

after(async () => {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
});

async function fixture(t, options = {}, blockedStorage = false) {
  const context = await browser.newContext({viewport:{width:1280,height:900}, colorScheme:'light', ...options, permissions:[], acceptDownloads:false});
  t.after(() => context.close());
  const external = [], errors = [];
  await context.route('**/*', route => {
    if (new URL(route.request().url()).origin !== base) { external.push('external request blocked'); return route.abort(); }
    return route.continue();
  });
  if (blockedStorage) await context.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {get() { throw new DOMException('Blocked', 'SecurityError'); }});
  });
  const page = await context.newPage();
  page.setDefaultTimeout(10000);
  page.setDefaultNavigationTimeout(15000);
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${new URL(response.url()).pathname}`); });
  t.after(() => { assert.deepEqual(external, []); assert.deepEqual(errors, []); });
  return page;
}

async function ready(page, hash = '') {
  await page.goto(base + '/' + hash);
  await page.locator('#interactive:visible').waitFor();
  await page.locator('#appearance:not([disabled])').waitFor();
}
async function fits(page) { assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No horizontal document overflow'); }
async function capture(page, name) {
  if (!process.env.RUNTIME_SCREENSHOT_DIR) return;
  await mkdir(process.env.RUNTIME_SCREENSHOT_DIR, {recursive:true});
  await page.screenshot({path:path.join(process.env.RUNTIME_SCREENSHOT_DIR, name + '.png'), fullPage:true});
}

test('all six collections use real retained values, boundaries and source links', async t => {
  const page = await fixture(t);
  await ready(page);
  assert.equal(await page.locator('#experiment').inputValue(), 'reuse');
  assert.equal(await page.locator('#condition').inputValue(), '1024:uniform');
  for (const study of data.experiments) {
    await page.locator('#experiment').selectOption(study.id);
    for (const condition of study.conditions) {
      await page.locator('#condition').selectOption(condition.id);
      const expected = study.rows.filter(row => row.condition === condition.id);
      const actual = await page.locator('#selection-table tbody tr').evaluateAll(rows => rows.map(row => ({method:row.dataset.method, text:row.textContent})));
      assert.deepEqual(actual.map(row => row.method), expected.map(row => row.method));
      expected.forEach((row, i) => { assert.ok(actual[i].text.includes(row.median_ms.toFixed(5))); assert.ok(actual[i].text.includes(row.first_ms.toFixed(5))); });
    }
    assert.equal(await page.locator('#boundary').textContent(), study.boundary);
    assert.equal(await page.locator('#excluded').textContent(), study.excluded);
    assert.equal(await page.locator('#source-links a').first().getAttribute('href'), study.source_url);
    assert.equal(await page.locator('#table-link').getAttribute('href'), '#table-' + study.id);
  }
  await page.locator('#experiment').selectOption('reuse');
  await page.locator('#condition').selectOption('1024:uniform');
  await page.locator('#metric').selectOption('kernel_ms');
  assert.equal(await page.locator('#selection-table tr[data-method^="cpu_"]').count(), 2);
  assert.match(await page.locator('#selection-table tr[data-method="cpu_optimized_1"]').textContent(), /Not measured/);
  assert.match(await page.locator('#selection-status').textContent(), /do not represent complete-call time/);
  await page.locator('#experiment').selectOption('inference-cuda');
  assert.equal(await page.locator('#metric').inputValue(), 'median_ms');
  assert.equal(await page.locator('#metric option[value="kernel_ms"]').isDisabled(), true);
  assert.match(await page.locator('#finding').textContent(), /CPU baseline was faster at all three/);
});

test('Auto and overrides preserve data through system changes and reload', async t => {
  const page = await fixture(t, {colorScheme:'dark'});
  await ready(page, '#experiment=inference-cuda&condition=8&metric=median_ms');
  const background = () => page.locator('body').evaluate(node => getComputedStyle(node).backgroundColor);
  assert.equal(await background(), 'rgb(9, 9, 9)');
  const before = await page.locator('#selection-table').textContent();
  await page.locator('#appearance').selectOption('clair');
  assert.equal(await background(), 'rgb(248, 247, 243)');
  await page.emulateMedia({colorScheme:'dark'});
  assert.equal(await background(), 'rgb(248, 247, 243)');
  await page.reload();
  await page.locator('#interactive:visible').waitFor();
  assert.equal(await page.locator('#appearance').inputValue(), 'clair');
  assert.equal(await page.locator('#selection-table').textContent(), before);
  await page.locator('#appearance').selectOption('obscur');
  assert.equal(await background(), 'rgb(9, 9, 9)');
  await page.locator('#appearance').selectOption('auto');
  await page.emulateMedia({colorScheme:'light'});
  assert.equal(await background(), 'rgb(248, 247, 243)');
  assert.equal(await page.evaluate(k => localStorage.getItem(k), key), null);
  assert.equal(await page.locator('#selection-table').textContent(), before);
});

test('blocked storage keeps appearance and comparison usable', async t => {
  const page = await fixture(t, {viewport:{width:390,height:844}}, true);
  await ready(page);
  await page.locator('#condition').selectOption('32:single_bin');
  await page.locator('#appearance').selectOption('obscur');
  assert.equal(await page.locator('#condition').inputValue(), '32:single_bin');
  assert.equal(await page.locator('html').getAttribute('data-appearance'), 'obscur');
  await page.reload();
  await page.locator('#interactive:visible').waitFor();
  assert.equal(await page.locator('#appearance').inputValue(), 'auto');
  assert.equal(await page.locator('#condition').inputValue(), '32:single_bin');
  await fits(page);
});

test('no JavaScript retains complete tables, memory and incorrect predictions', async t => {
  const page = await fixture(t, {javaScriptEnabled:false, viewport:{width:390,height:844}, colorScheme:'dark'});
  await page.goto(base + '/');
  assert.equal(await page.locator('#interactive').isVisible(), false);
  assert.equal(await page.locator('#appearance').isDisabled(), true);
  assert.equal(await page.locator('#tables details').count(), 6);
  assert.equal(await page.locator('#tables tbody tr').count(), 123 + 36);
  await page.locator('#table-inference-cuda summary').click();
  assert.match(await page.locator('#table-inference-cuda').textContent(), /CPU baseline was faster at all three/);
  assert.match(await page.locator('#inference').textContent(), /Misclassified/);
  assert.equal(await page.locator('#inference tbody tr').count(), 8 + 80);
  await page.locator('#memory summary').click();
  assert.equal(await page.locator('#memory tbody tr').count(), 16);
  await fits(page);
  await capture(page, 'runtime-no-js-390-obscur');
});

for (const [width, appearance] of [[1280,'clair'], [390,'obscur']]) test(`comparison is readable at ${width} in ${appearance}`, async t => {
  const page = await fixture(t, {viewport:{width,height:width === 390 ? 844 : 900}, hasTouch:width === 390});
  await ready(page);
  await page.locator('#appearance').selectOption(appearance);
  await fits(page);
  assert.equal(await page.locator('#chart .bar-row').count(), 6);
  await page.evaluate(() => document.fonts.ready);
  if ((process.env.RUNTIME_BROWSER_ENGINE ?? 'chromium') === 'chromium') {
    const session = await page.context().newCDPSession(page);
    try {
      await session.send('DOM.enable');
      await session.send('CSS.enable');
      const document = await session.send('DOM.getDocument');
      const {nodeId} = await session.send('DOM.querySelector', {nodeId:document.root.nodeId, selector:'h1'});
      const {fonts} = await session.send('CSS.getPlatformFontsForNode', {nodeId});
      const providers = fonts.filter(font => font.glyphCount > 0);
      assert.ok(providers.length > 0, 'Actual heading glyph supplier recorded');
      console.log('RUNTIME_FONT_EVIDENCE ' + JSON.stringify({width, appearance,
        cssFamily:await page.locator('h1').evaluate(node => getComputedStyle(node).fontFamily), providers}));
    } finally { await session.detach(); }
  }
  await page.locator('#table-link').click();
  assert.equal(await page.locator('#table-reuse').evaluate(node => node.open), true);
  await page.locator('#table-reuse summary').click();
  assert.equal(await page.locator('#table-reuse').evaluate(node => node.open), false);
  await page.locator('#table-link').click();
  assert.equal(await page.locator('#table-reuse').evaluate(node => node.open), true, 'Same-anchor link reopens a manually closed table');
  await fits(page);
  await capture(page, `runtime-${width}-${appearance}`);
});

test('keyboard and doubled phone text keep the comparison and table reachable', async t => {
  const page = await fixture(t, {viewport:{width:320,height:700}, reducedMotion:'reduce'});
  await ready(page);
  await page.addStyleTag({content:'body{font-family:"Runtime deliberately missing face",Arial,sans-serif!important}'});
  const doubleText = () => page.evaluate(() => {
    const saved = window.__runtimeOriginalTextSizes ??= new WeakMap();
    const nodes = [...document.querySelectorAll('h1,h2,h3,h4,p,a,label,select,summary,th,td,.bar-row>span')];
    // Restore only fixture font overrides before measuring newly created nodes.
    // This avoids doubling an inherited size whose parent was already enlarged.
    nodes.forEach(node => { if (saved.has(node)) node.style.fontSize = saved.get(node).inline; });
    const sizes = nodes.map(node => saved.get(node) ?? {inline:node.style.fontSize, pixels:parseFloat(getComputedStyle(node).fontSize)});
    nodes.forEach((node, i) => { saved.set(node, sizes[i]); node.style.fontSize = `${sizes[i].pixels*2}px`; });
  });
  await doubleText();
  await page.locator('#experiment').focus();
  // The retained closed-select ArrowDown/Enter/Tab trace opened the popup, then
  // Tab closed it without advancing focus. Open, choose, confirm, then advance.
  const snapshot = async (stage, key) => {
    const state = await page.evaluate(() => {
      const active = document.activeElement;
      let popupOpen = null;
      try { popupOpen = active?.matches(':open') ?? false; } catch { /* Unsupported selector. */ }
      return {activeId:(active?.id ?? '').slice(0,80), activeTag:active?.tagName ?? null,
        activeValue:active instanceof HTMLSelectElement ? active.value.slice(0,80) : null,
        selectedIndex:active instanceof HTMLSelectElement ? active.selectedIndex : null, popupOpen,
        experiment:document.getElementById('experiment').value.slice(0,80),
        condition:document.getElementById('condition').value.slice(0,80), documentFocused:document.hasFocus()};
    });
    console.log('RUNTIME_KEYBOARD_DIAGNOSTIC ' + JSON.stringify({stage, key, ...state}));
  };
  for (const key of ['Enter', 'ArrowDown', 'Enter', 'Tab']) {
    await snapshot('before', key);
    await page.keyboard.press(key);
    await snapshot('after', key);
  }
  assert.equal(await page.locator('#experiment').inputValue(), 'inference-cpu', 'Keyboard selected the next retained experiment');
  assert.equal(await page.locator('#condition').evaluate(node => node === document.activeElement), true);
  await doubleText();
  await page.locator('#selection-table .table-wrap').focus();
  assert.equal(await page.locator('#selection-table .table-wrap').evaluate(node => node === document.activeElement), true);
  await fits(page);
  await capture(page, 'runtime-320-doubled-fallback');
});

test('a synthetic malformed embedded record exposes the static recovery path', async t => {
  const page = await fixture(t);
  await page.route(base + '/', async route => {
    const html = await readFile(path.join(root, 'index.html'), 'utf8');
    const changed = html.replace(/(<script id="evidence-data" type="application\/json">)[\s\S]*?(<\/script>)/, '$1{broken$2');
    assert.notEqual(changed, html, 'Fixture changed only the embedded record');
    await route.fulfill({status:200, contentType:'text/html', body:changed});
  });
  await page.goto(base + '/');
  await page.locator('#explorer-error:visible').waitFor();
  assert.equal(await page.locator('#interactive').isVisible(), false);
  assert.equal(await page.locator('#tables tbody tr').count(), 123 + 36);
  await page.locator('#table-reuse summary').click();
  assert.match(await page.locator('#table-reuse').textContent(), /0\.27150/);
});
