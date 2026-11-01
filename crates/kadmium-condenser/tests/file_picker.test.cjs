// Exercise the real native bridge with a simulated OS response; Rust tests cover path handling.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(path.resolve('crates/kadmium-gateway/tests/ui/node_modules/playwright'));

test('native pickers preserve cancel, use workspace checks, and expose an explicit embedded fallback', { timeout: 60000 }, async () => {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'condenser-picker-'));
  fs.writeFileSync(path.join(cwd, 'hello.txt'), 'Workspace fixture');
  fs.writeFileSync(path.join(cwd, '.env'), 'sensitive fixture');
  const server = spawn(path.resolve('target/debug/kadmium-condenser' + (process.platform === 'win32' ? '.exe' : '')), ['serve', '--port', '0'], {
    cwd, windowsHide: true, env: { ...process.env, CONDENSER_HOME: path.join(cwd, 'home'), KADMIUM_MODEL_LIBRARY: path.join(cwd, 'models') }
  });
  let browser;
  try {
    const base = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(Error('Server startup timed out')), 20000);
      server.once('error', error => { clearTimeout(timer); reject(error); });
      server.once('exit', code => { clearTimeout(timer); reject(Error('Server exited: ' + code)); });
      server.stdout.on('data', chunk => { const match = String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/); if (match) { clearTimeout(timer); resolve(match[0]); } });
    });
    browser = await chromium.launch({ channel: process.env.KADMIUM_TEST_BROWSER || (process.platform === 'win32' ? 'msedge' : undefined), headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      window.pickerRequests = [];
      window.ipc = { postMessage(text) { window.pickerRequests.push(JSON.parse(text)); } };
    });
    await page.addInitScript({ content: fs.readFileSync(path.resolve('crates/kadmium-native/src/bridge.js'), 'utf8').replace('__KADMIUM_ORIGIN__', JSON.stringify(base)) });
    await page.goto(base + '/#models');
    const respond = response => page.evaluate(value => {
      const request = window.pickerRequests.at(-1);
      window.__KADMIUM_FILES__.complete({ id: request.id, ...value });
    }, response);
    await page.locator('#checkpoint-path').fill(cwd);
    await page.locator('#checkpoint-browse').click();
    assert.equal(await page.locator('#checkpoint-browser').isVisible(), false);
    assert.equal(await page.evaluate(() => window.pickerRequests.at(-1).kind), 'model_file');
    await respond({ path: null });
    assert.equal(await page.locator('#checkpoint-path').inputValue(), cwd);
    await page.locator('#checkpoint-browse-folder').click();
    assert.equal(await page.evaluate(() => window.pickerRequests.at(-1).kind), 'model_folder');
    await respond({ path: path.join(cwd, 'models') });
    assert.equal(await page.locator('#checkpoint-path').inputValue(), path.join(cwd, 'models'));
    await page.locator('#checkpoint-path').fill(cwd);
    await page.locator('#checkpoint-embedded').click();
    await page.locator('#checkpoint-choose-folder').waitFor({ state: 'visible' });
    await page.locator('#checkpoint-browser-close').click();

    // Workspace files live beside the task, not on the Models page.
    await page.locator('.rail-nav[data-view="harness"]').click();
    await page.locator('[data-context-tab="files"]').click();
    await page.waitForFunction(() => document.querySelector('#workspace-name').title !== '');
    assert.equal(await page.locator('.files-browser').isVisible(), false);
    await page.locator('#files-open-native').click();
    assert.equal(await page.evaluate(() => window.pickerRequests.at(-1).kind), 'workspace_file');
    await respond({ path: 'hello.txt' });
    await page.waitForFunction(() => document.querySelector('#file-preview-content').textContent === 'Workspace fixture');
    assert.equal(await page.locator('.files-browser').isVisible(), false);
    const widths = await page.evaluate(() => ({
      editor: document.querySelector('#file-preview').getBoundingClientRect().width,
      workspace: document.querySelector('.file-workspace').getBoundingClientRect().width
    }));
    assert.ok(widths.editor >= widths.workspace - 2, 'native selection gives the editor the full dock width');
    await page.locator('#files-open-native').click();
    await respond({ path: null });
    assert.equal(await page.locator('#preview-file-name').textContent(), 'hello.txt');
    await page.locator('#files-open-native').click();
    await respond({ error: 'Select a file inside the active workspace.' });
    await page.waitForFunction(() => document.querySelector('#file-operation-status').textContent.includes('inside the active workspace'));
    await page.locator('#files-open-native').click();
    await respond({ path: '.env' });
    await page.waitForFunction(() => document.querySelector('#edit-file').disabled);
    assert.notEqual(await page.locator('#file-preview-content').textContent(), 'sensitive fixture');
    await page.locator('#close-file-preview').click();
    await page.locator('#files-toggle-tree').click();
    await page.locator('.file-row', { hasText: 'hello.txt' }).waitFor();
    assert.equal(await page.locator('.files-browser').isVisible(), true);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    }
    assert.deepEqual(errors, []);
  } finally {
    if (browser) await browser.close();
    server.kill();
    await new Promise(resolve => server.exitCode === null ? server.once('exit', resolve) : resolve());
    fs.rmSync(cwd, { recursive: true, force: true });
  }
});
