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
    // Permanent HTTP errors stay manual; temporary failures back off and stop.
    for (const status of [404, 503]) {
      const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
      const panel = await context.newPage(); let attempts = 0;
      await panel.clock.install();
      const endpoint = '/api/v1/training/schedule';
      await panel.route('**' + endpoint, route => {
        attempts++;
        return route.fulfill({ status, contentType: 'application/json', body: '{"error":"unavailable"}' });
      });
      await panel.goto(base + '/dashboard#/training');
      await panel.getByRole('button', { name: 'Retry failed panels' }).waitFor();
      if (status === 503) {
        for (const delay of [3000, 6000, 12000]) {
          const response = panel.waitForResponse(r => r.url().endsWith(endpoint));
          await panel.clock.fastForward(delay);
          await response;
          await panel.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
        }
      }
      await panel.clock.fastForward(120000);
      assert.equal(attempts, status === 404 ? 1 : 4, 'automatic retry budget');
      assert.doesNotMatch(await panel.locator('body').innerText(), /reconnecting/);
      await panel.unroute('**' + endpoint);
      await panel.getByRole('button', { name: 'Retry failed panels' }).click();
      await panel.waitForFunction(() => !document.body.innerText.includes('Some panels could not load'));
      await context.close();
    }
    // Polled panels recover after outages longer than the initial 3/6/12-second budget.
    for (const failure of ['network', '503', '404']) {
      const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
      const panel = await context.newPage();
      panel.on('pageerror', e => errors.push(e.message));
      await panel.clock.install();
      const endpoints = ['/api/v1/cluster/overview', '/api/v1/cluster/workers'];
      const attempts = Object.fromEntries(endpoints.map(endpoint => [endpoint, 0]));
      let recovered = false;
      for (const endpoint of endpoints) {
        await panel.route('**' + endpoint, async route => {
          attempts[endpoint]++;
          if (recovered) {
            const response = await route.fetch();
            const body = await response.json();
            if (endpoint.endsWith('/workers')) body.live_worker_count = 7;
            else body.coordinator_quorum.active_leader = 'recovered-after-outage';
            return route.fulfill({ response, json: body });
          }
          if (failure === 'network') return route.abort('failed');
          return route.fulfill({ status: Number(failure), contentType: 'application/json', body: '{"error":"unavailable"}' });
        });
      }
      await panel.goto(base + '/dashboard');
      await panel.getByRole('button', { name: 'Retry failed panels' }).waitFor();
      const tick = async delay => {
        const done = endpoints.map(endpoint => failure === 'network'
          ? panel.waitForEvent('requestfailed', request => request.url().endsWith(endpoint))
          : panel.waitForResponse(response => response.url().endsWith(endpoint)));
        await panel.clock.fastForward(delay);
        await Promise.all(done);
        await panel.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
      };
      if (failure !== '404') {
        for (const delay of [3000, 6000, 12000]) await tick(delay);
        for (let i = 0; i < 12; i++) await tick(5000);
        for (const endpoint of endpoints) assert.ok(attempts[endpoint] > 4, endpoint + ' stopped polling');
      } else {
        await panel.clock.fastForward(120000);
        for (const endpoint of endpoints) assert.equal(attempts[endpoint], 1, endpoint + ' retried permanent error');
      }
      recovered = true;
      if (failure === '404') await panel.getByRole('button', { name: 'Retry failed panels' }).click();
      else await panel.clock.fastForward(5000);
      await panel.getByText('7 live workers', { exact: true }).waitFor();
      await panel.locator('.cr-top [title*="recovered-after-outage"]').waitFor();
      await panel.waitForFunction(() => !document.body.innerText.includes('Some panels could not load'));
      await context.close();
    }
    // Force an actual journal replacement failure in the isolated fixture process.
    const journal = path.join(cwd, 'target/kadmium-state/cluster_state.demo.json');
    const savedJournal = fs.readFileSync(journal);
    const journalFiles = fs.readdirSync(path.dirname(journal));
    fs.renameSync(journal, journal + '.backup');
    fs.mkdirSync(journal);
    const before = await (await page.request.get(base + '/api/v1/cluster/overview')).json();
    for (const [endpoint, data] of [
      ['/api/v1/cluster/model', { inference_model_id: 'qwen2.5-coder-32b-instruct' }],
      ['/api/v1/models/import', { model_id: 'qwen2.5-coder-32b-instruct', demo_fixture: true }],
      ['/api/v1/storage/pool-bucket/object', { object_uri: 'failure-test', logical_size_gb: 1 }],
      ['/api/v1/storage/pool-bucket/anchor', { anchor_uri: 'nas://failure-test' }],
    ]) {
      const response = await page.request.post(base + endpoint, { data });
      assert.equal(response.status(), 500, endpoint + ': ' + await response.text());
      const after = await (await page.request.get(base + '/api/v1/cluster/overview')).json();
      assert.equal(after.epoch, before.epoch);
      assert.equal(after.inference_plan.model_id, before.inference_plan.model_id);
      assert.deepEqual(after.imported_models, before.imported_models);
    }
    fs.rmdirSync(journal);
    fs.renameSync(journal + '.backup', journal);
    assert.deepEqual(fs.readFileSync(journal), savedJournal);
    assert.deepEqual(fs.readdirSync(path.dirname(journal)), journalFiles);
    const companion = await browser.newPage();
    companion.on('pageerror', e => errors.push(e.message));
    await companion.goto(base + '/companion#/models');
    await companion.getByText(/The catalog below creates a placement plan/).waitFor();
    assert.doesNotMatch(await companion.locator('body').innerText(), /Running now|No model loaded|Load one and Kadmium/);
    await companion.close();
    assert.deepEqual(errors, []);
  } finally {
    if (browser) await browser.close();
    server.kill();
    await new Promise(resolve => { if (server.exitCode !== null) resolve(); else server.once('exit', resolve); });
    fs.rmSync(cwd, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
