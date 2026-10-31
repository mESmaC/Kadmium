// Uses the same Playwright installation and built binary as local_compute.test.cjs.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(path.resolve('crates/kadmium-gateway/tests/ui/node_modules/playwright'));

test('agent controls persist, built-ins are selectable, and request settings match the task', { timeout: 60000 }, async () => {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'condenser-agent-ui-'));
  const server = spawn(path.resolve('target/debug/kadmium-condenser' + (process.platform === 'win32' ? '.exe' : '')), ['serve', '--port', '0'], {
    cwd, windowsHide: true, env: { ...process.env, CONDENSER_HOME: path.join(cwd, 'home'), KADMIUM_MODEL_LIBRARY: path.join(cwd, 'models') }
  });
  let browser;
  try {
    const base = await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.stdout.on('data', chunk => { const match = String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/); if (match) resolve(match[0]); });
      server.once('exit', code => reject(new Error('server exited: ' + code)));
    });
    browser = await chromium.launch({ channel: process.env.KADMIUM_TEST_BROWSER || (process.platform === 'win32' ? 'msedge' : undefined), headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.locator('[data-context-tab="run"]').click();
    assert.equal(await page.locator('#research-required').isChecked(), false);
    assert.equal(await page.locator('#memory-enabled').isChecked(), false);
    await page.locator('#research-required').check();
    await page.locator('#memory-enabled').check();
    await page.locator('#save-instructions').click();
    await page.reload();
    await page.locator('[data-context-tab="run"]').click();
    assert.equal(await page.locator('#research-required').isChecked(), true);
    assert.equal(await page.locator('#memory-enabled').isChecked(), true);
    const source = page.locator('.tool-choice').filter({ hasText: 'condenser__file_read' }).locator('input');
    await source.check();
    assert.equal(await page.locator('.tool-choice').count(), 12);
    await page.locator('label').filter({ has: page.locator('#enable-tools') }).click();
    let payload;
    await page.route('**/api/v1/chat/stream', async route => {
      payload = route.request().postDataJSON();
      const answer = { role: 'assistant', content: 'Verified fixture.', tool_calls: [], is_error: false };
      const report = { provider: payload.provider.kind, model: payload.provider.model, status: 'completed', answer, messages: [...payload.messages, answer], steps: [], diagnostics: [], usage: {} };
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: ' + JSON.stringify({ event: 'finished', report }) + '\n\n' });
    });
    await page.locator('#prompt').fill('Inspect a source');
    await page.locator('#send').click();
    await page.waitForFunction(() => document.querySelector('#task-status').textContent === 'Complete');
    assert.equal(payload.agent.research_required, true);
    assert.match(payload.agent.session_id, /^[\w-]+$/);
    assert.deepEqual(payload.allowed_tools, ['condenser__file_read']);
    assert.equal(payload.agent.allow_host_commands, false);
    assert.equal(payload.agent.operating_instructions, true);
    assert.ok(await page.locator('.tool-choice').filter({ hasText: 'condenser__command · side effects' }).count());
    assert.equal(payload.provider.timeout_secs, 0);
    assert.equal(payload.sampling.max_tokens == null, true);
    assert.equal(payload.sampling.context_window == null, true);
    await page.locator('#fork-task').click();
    assert.equal(await page.locator('#research-required').isChecked(), true);
    assert.equal(await page.locator('#memory-enabled').isChecked(), true);
    fs.mkdirSync(path.resolve('target/agent-smoke'), { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator('[data-context-tab="run"]').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      await page.locator('#research-required').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.resolve('target/agent-smoke/run-' + width + '.png') });
    }
    assert.deepEqual(errors, []);
  } finally {
    if (browser) await browser.close();
    server.kill();
    await new Promise(resolve => server.exitCode === null ? server.once('exit', resolve) : resolve());
    fs.rmSync(cwd, { recursive: true, force: true });
  }
});
