// Build: cargo build -p kadmium-gateway --example control_room_fixture --locked
// Run: node --test crates/kadmium-gateway/tests/model_setup.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(path.resolve('crates/kadmium-gateway/tests/ui/node_modules/playwright'));

test('Control Room and desktop Companion launch the shared preparation flow and retain quality settings', { timeout: 120000 }, async () => {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kadmium-model-setup-'));
  const source = path.join(cwd, 'model'); fs.mkdirSync(source);
  fs.writeFileSync(path.join(source, 'config.json'), JSON.stringify({ model_type: 'llama', max_position_embeddings: 131072 }));
  fs.writeFileSync(path.join(source, 'tokenizer.json'), '{}');
  const header = Buffer.from(JSON.stringify({ weight: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } }));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(path.join(source, 'model.safetensors'), Buffer.concat([length, header, Buffer.from([0, 0, 128, 63])]));
  const server = spawn(path.resolve('target/debug/examples/control_room_fixture' + (process.platform === 'win32' ? '.exe' : '')), [], {
    cwd, windowsHide: true, env: { ...process.env, KADMIUM_MODEL_LIBRARY: path.join(cwd, 'library'), KADMIUM_CONDENSER_WORKSPACE: cwd }
  });
  let browser;
  try {
    // The fixture prints this launch's Condenser access link.
    const access = await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.stdout.on('data', chunk => { const match = String(chunk).match(/http:\/\/127\.0\.0\.1:\d+\S*/); if (match) resolve(match[0]); });
      server.once('exit', code => reject(new Error('fixture exited: ' + code)));
    });
    const base = new URL(access).origin;
    browser = await chromium.launch({ channel: process.env.KADMIUM_TEST_BROWSER || (process.platform === 'win32' ? 'msedge' : undefined), headless: true });
    for (const shell of ['/#/models', '/companion#models']) {
      const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
      const errors = []; page.on('pageerror', e => errors.push(e.message));
      if (shell.startsWith('/companion')) await page.addInitScript(() => {
        window.__KADMIUM_DESKTOP__ = { platform: 'windows', version: 'test', openExternal() { throw new Error('Condenser should stay in the desktop window'); } };
      });
      const locked = await page.goto(base + '/condenser/');
      assert.equal(locked.status(), 401);
      await page.goto(access);
      await page.goto(base + shell);
      await page.getByRole('heading', { name: 'Run your own model' }).waitFor();
      assert.equal(await page.getByRole('link', { name: /Send to Kadmium Cloud/ }).getAttribute('href'), '/condenser/#models-cloud');
      await page.getByRole('link', { name: /Prepare a local model/ }).click();
      await page.waitForURL('**/condenser/#models');
      await page.locator('#return-kadmium').waitFor();
      assert.equal(await page.locator('#compute-url').inputValue(), base);
      await page.locator('#checkpoint-path').fill(source);
      await page.locator('#checkpoint-browse').click();
      await page.getByRole('button', { name: 'Model: model.safetensors', exact: true }).click();
      assert.ok((await page.locator('#checkpoint-path').inputValue()).endsWith('model'));
      await page.locator('#checkpoint-import').click();
      await page.waitForFunction(() => document.querySelector('#checkpoint-status').textContent.includes('Normalized 1 tensors'));
      const id = await page.locator('#checkpoint-select').inputValue();
      assert.match(id, /^[a-f0-9]{64}$/);
      assert.match(await page.locator('#checkpoint-directory').textContent(), /library/);
      assert.ok(fs.existsSync(path.join(cwd, 'library', id, 'banks', 'bank-00000.ktb')));
      await page.route('**/condenser/api/v1/checkpoints/load', route => {
        assert.equal(route.request().postDataJSON().context_length, null);
        return route.fulfill({ json: { checkpoint: { id, name: 'model', context_length: 131072 }, context_length: 131072, provider: { kind: 'ollama', model: 'kadmium-' + id + ':latest', base_url: 'http://127.0.0.1:11434' }, execution: 'Fixture backend' } });
      });
      await page.locator('#checkpoint-load').click();
      await page.waitForFunction(() => !document.querySelector('#checkpoint-use').disabled);
      await page.locator('#checkpoint-use').click();
      await page.locator('#close-context').click();
      await page.locator('#provider-button').click();
      assert.equal(await page.locator('#max-tokens').inputValue(), '');
      assert.equal(await page.locator('#context-window').inputValue(), '');
      assert.equal(await page.locator('#provider-timeout').inputValue(), '');
      await page.locator('#max-tokens').fill('4096');
      await page.locator('#context-window').fill('32768');
      await page.locator('#provider-timeout').fill('900');
      await page.locator('#save-settings').click();
      await page.reload();
      await page.locator('#view-models').waitFor();
      await page.locator('#provider-button').click();
      assert.equal(await page.locator('#max-tokens').inputValue(), '4096');
      assert.equal(await page.locator('#context-window').inputValue(), '32768');
      assert.equal(await page.locator('#provider-timeout').inputValue(), '900');
      await page.locator('#quality-maximum').click();
      await page.locator('#save-settings').click();
      await page.locator('#open-model-library').click();
      for (const width of [1440, 940, 390]) {
        await page.setViewportSize({ width, height: 1000 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      }
      await page.setViewportSize({ width: 1440, height: 1000 });
      fs.mkdirSync(path.resolve('target/checkpoint-smoke'), { recursive: true });
      await page.screenshot({ path: path.resolve('target/checkpoint-smoke/' + (shell.startsWith('/companion') ? 'desktop' : 'control-room') + '-model-setup.png') });
      assert.deepEqual(errors, []);
      await page.close();
    }
  } finally {
    if (browser) await browser.close();
    server.kill(); await new Promise(resolve => server.exitCode === null ? server.once('exit', resolve) : resolve());
    // cwd is an owned, freshly-created OS temporary directory.
    fs.rmSync(cwd, { recursive: true, force: true });
  }
});
