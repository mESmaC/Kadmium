// Setup: npm install --prefix target/ui-test --no-package-lock playwright@1.64.1
// Build: cargo build -p kadmium-gateway --example control_room_fixture
// Run: node --test crates/kadmium-gateway/tests/control_room.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(path.resolve('target/ui-test/node_modules/playwright'));

test('Control Room mobile navigation, recovery, and honest diagnostic status', { timeout: 120000 }, async () => {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kadmium-ui-'));
  // The router embeds assets; a disposable cwd isolates any journal writes.
  const server = spawn(path.resolve('target/debug/examples/control_room_fixture' + (process.platform === 'win32' ? '.exe' : '')), [], { cwd, windowsHide: true });
  let browser;
  try {
    const base = await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.stdout.on('data', chunk => { const match = String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/); if (match) resolve(match[0]); });
      server.once('exit', code => reject(new Error('fixture exited: ' + code)));
    });
    browser = await chromium.launch({ channel: process.env.KADMIUM_TEST_BROWSER || (process.platform === 'win32' ? 'msedge' : 'chromium'), headless: true });
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto(base + '/dashboard');
    await page.getByRole('button', { name: 'Open navigation', exact: true }).waitFor();
    await page.getByRole('link', { name: 'Skip to content' }).focus();
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('aria-label')), 'Open navigation');
    assert.equal(await page.locator('#cr-side').getAttribute('aria-hidden'), 'true');
    assert.equal(await page.locator('#cr-side').evaluate(el => el.inert), true);
    const open = () => page.getByRole('button', { name: 'Open navigation', exact: true }).click();
    await open();
    await page.waitForFunction(() => document.querySelector('#cr-side').contains(document.activeElement));
    assert.equal(await page.locator('#cr-side').getAttribute('aria-modal'), 'true');
    await page.keyboard.press('Shift+Tab');
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('href')), '/metrics');
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Close navigation');
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => document.activeElement.getAttribute('aria-label') === 'Open navigation');
    await open();
    await page.locator('.cr-scrim').click({ position: { x: 350, y: 400 } });
    await page.waitForFunction(() => document.activeElement.getAttribute('aria-label') === 'Open navigation');
    await open();
    await page.locator('#cr-side a[href="#/models"]').click();
    await page.waitForFunction(() => document.querySelector('#cr-side').inert);
    assert.equal(await page.locator('.cr-menu[aria-controls]').getAttribute('aria-expanded'), 'false');
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.waitForFunction(() => !document.querySelector('#cr-side').inert);
    assert.equal(await page.locator('#cr-side').getAttribute('aria-hidden'), null);
    await page.locator('#cr-side a[href="#/models"]').focus();
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('href')), '#/models');

    await page.getByRole('button', { name: /llama-3.2-3b-instruct/ }).click();
    await page.getByText('planned', { exact: true }).first().waitFor();
    assert.match(await page.locator('main').innerText(), /not a generated model answer/);
    assert.doesNotMatch(await page.locator('main').innerText(), /Weights in memory|loads the best fit first|load it/);
    await page.getByRole('button', { name: 'Compile reference fixture', exact: true }).click();
    await page.getByText('synthetic fixture', { exact: true }).waitFor();
    await page.locator('#cr-side a[href="#/gateway"]').click();
    await page.getByText(/The selector labels the response only/).waitFor();
    await page.getByRole('button', { name: 'Send', exact: true }).click();
    await page.locator('.cr-reply').waitFor();
    await page.locator('#cr-side a[href="#/data"]').click();
    await page.getByText('durability unverified', { exact: true }).waitFor();
    assert.doesNotMatch(await page.locator('main').innerText(), /rebuild test passed|half the machines lost: survives/);

    // New contexts ensure each endpoint's initial request fails without a warm cache.
    for (const endpoint of ['/api/v1/training/schedule', '/api/v1/cluster/overview']) {
      for (const failure of ['network', 'http', 'json']) {
        const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
        const panel = await context.newPage(); let attempts = 0;
        panel.on('pageerror', e => errors.push(e.message));
        await panel.route('**' + endpoint, async route => {
          attempts++;
          if (attempts > 1) return route.continue();
          if (failure === 'network') return route.abort('failed');
          return route.fulfill({ status: failure === 'http' ? 500 : 200, contentType: 'application/json', body: failure === 'json' ? '{broken' : '{"error":"failed"}' });
        });
        await panel.goto(base + '/dashboard#/training');
        await panel.getByRole('button', { name: 'Retry failed panels' }).waitFor();
        if (failure === 'http') await panel.getByRole('button', { name: 'Retry failed panels' }).click();
        await panel.waitForFunction(() => !document.body.innerText.includes('Some panels could not load'), { timeout: 12000 });
        assert.ok(attempts >= 2, `${endpoint}: ${failure} did not retry`);
        await panel.waitForFunction(() => !document.body.innerText.includes('Loading schedule'), { timeout: 12000 });
        await context.close();
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    if (browser) await browser.close();
    server.kill();
    await new Promise(resolve => { if (server.exitCode !== null) resolve(); else server.once('exit', resolve); });
    fs.rmSync(cwd, { recursive: true, force: true });
  }
});
