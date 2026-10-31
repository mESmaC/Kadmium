const { test: base, expect } = require('@playwright/test');
const { readFileSync } = require('node:fs');
const fs = require('node:fs/promises');
const path = require('node:path');

const test = base.extend({ sitePath: ['/site', { option: true }] });

test('marketing: product boundaries and local setup remain readable without JavaScript', async ({ browser, baseURL, sitePath }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  try {
    await page.goto(new URL(sitePath, baseURL).href);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Your hardware.');
    for (const name of ['Control Room', 'Companion', 'Condenser']) {
      await expect(page.locator('#product').getByRole('heading', { name, exact: true })).toBeVisible();
    }
    await expect(page.locator('#quality')).toContainText('Leave quality overrides blank for maximum');
    await expect(page.locator('#cloud')).toContainText('not services you can purchase');
    await expect(page.locator('#commands')).toContainText('cargo run --locked -p kadmium-cli -- serve --port 7848');
    await expect(page.getByRole('link', { name: 'Self-host', exact: true })).toBeVisible();
    await page.getByText('Can I earn money by hosting a node?', { exact: true }).click();
    await expect(page.getByText('There is no live paid Supply Grid', { exact: false })).toBeVisible();
    await expect(page.locator('#theme')).toBeHidden();
    await expect(page.locator('#copy')).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  } finally { await context.close(); }
});

test('marketing: responsive page, assets, anchors and documentation destinations', async ({ page, sitePath }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()}: ${response.url()}`); });
  await page.goto(sitePath);
  await page.evaluate(() => document.fonts.ready);
  for (const width of [1440, 1180, 940, 768, 760, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow at ${width}px`).toBe(true);
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  }
  const missingAnchors = await page.locator('a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)).filter(id => !document.getElementById(id)));
  expect(missingAnchors).toEqual([]);
  const docLinks = await page.locator('a[href^="https://github.com/mESmaC/Kadmium/blob/main/"]').evaluateAll(links => links.map(link => link.href));
  expect(docLinks.length).toBeGreaterThan(0);
  for (const href of docLinks) {
    const destination = new URL(href);
    const relativePath = destination.pathname.slice('/mESmaC/Kadmium/blob/main/'.length);
    const document = readFileSync(path.resolve(__dirname, '../../../..', relativePath), 'utf8');
    if (destination.hash) {
      const headings = Array.from(document.matchAll(/^#{1,6} (.+)$/gm), match => match[1].trim().toLowerCase().replace(/[^a-z0-9 _-]/g, '').replaceAll(' ', '-'));
      expect(headings, href).toContain(destination.hash.slice(1));
    }
  }
  await expect(page.locator('#contact a[href^="mailto:"]')).toHaveAttribute('href', /mailto:hello@kadmium\.ai\?subject=/);
  expect(await page.locator('img').evaluateAll(images => images.every(img => img.complete && img.naturalWidth > 0))).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: 'test-results/marketing-dark.png', fullPage: true });
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect(page.locator('#wordmark')).toHaveAttribute('src', /wordmark-light\.svg$/);
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.screenshot({ path: 'test-results/marketing-light.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await page.screenshot({ path: 'test-results/marketing-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('marketing: public page stays separate from local APIs and cloud promises', async ({ page, baseURL, sitePath }) => {
  const requests = [];
  page.on('request', request => requests.push(new URL(request.url())));
  await page.goto(sitePath);
  await page.evaluate(() => document.fonts.ready);
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', 'https://kadmium.ai/');
  await expect(page.locator('#quality')).toContainText('not a guessed token cap');
  await expect(page.locator('#quality')).toContainText('Do not silently truncate prompts, compact history, quantize weights or switch models or backends');
  await expect(page.locator('#quality')).toContainText('Reset to maximum');
  await expect(page.locator('#cloud .tag')).toHaveText(['PLANNED', 'SIMULATED', 'PLANNED']);
  await expect(page.locator('a[href^="/docs"], a[href^="/condenser"], a[href^="/companion"], a[href^="/api"]')).toHaveCount(0);
  await page.getByText('Can I run my local workspace on this website?', { exact: true }).click();
  await expect(page.getByText('This website does not connect to your local coordinator', { exact: false })).toBeVisible();
  await page.getByText('Is a hosted provider the same as Kadmium Cloud?', { exact: true }).click();
  await expect(page.getByText('A provider you configure in Condenser is a separate service', { exact: false })).toBeVisible();
  expect(requests.length).toBeGreaterThan(0);
  for (const request of requests) {
    expect(request.origin).toBe(new URL(baseURL).origin);
    expect(request.pathname).not.toMatch(/^[/](api|v1|condenser|companion|docs)([/]|$)/);
  }
});

test('marketing: static export includes only the public page and shared assets', async ({}, testInfo) => {
  const { buildWebsite } = await import('../../../../scripts/build_website.mjs');
  const output = await buildWebsite(testInfo.outputPath('website'));
  expect(output.files.toSorted()).toEqual([
    'assets/Logos/kadmium-favicon.svg',
    'assets/Logos/kadmium-wordmark-dark.svg',
    'assets/Logos/kadmium-wordmark-light.svg',
    'fonts/Archivo-Variable.woff2',
    'fonts/JetBrainsMono-Variable.woff2',
    'index.html',
    'tokens.css'
  ]);
  for (const file of output.files) {
    const source = file === 'index.html' ? 'crates/kadmium-gateway/src/marketing_site.html' : 'design-system/' + file;
    const expected = await fs.readFile(path.resolve(__dirname, '../../../..', source));
    expect(await fs.readFile(path.join(output.directory, file)), file).toEqual(expected);
  }
  const occupied = testInfo.outputPath('occupied');
  await fs.writeFile(occupied, 'not a directory');
  await expect(buildWebsite(occupied)).rejects.toThrow();
});

test('marketing: keyboard navigation, mobile menu and resize recovery', async ({ page, sitePath }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(sitePath);
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();
  const menu = page.getByRole('button', { name: 'Menu', exact: true });
  await expect(menu).toHaveAttribute('aria-expanded', 'false');
  await menu.click();
  await expect(menu).toHaveAttribute('aria-expanded', 'true');
  await page.getByRole('link', { name: 'Self-host', exact: true }).click();
  await expect(page).toHaveURL(/#local$/);
  await expect(page.locator('#local')).toBeFocused();
  await expect(menu).toHaveAttribute('aria-expanded', 'false');
  await menu.click();
  await page.keyboard.press('Escape');
  await expect(menu).toBeFocused();
  await expect(page.locator('#navigation')).toBeHidden();
  await page.setViewportSize({ width: 1180, height: 900 });
  await expect(page.locator('#navigation')).toBeVisible();
  await expect(menu).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('#navigation')).toBeHidden();
});

test('marketing: copy commands and recover from denied clipboard or storage', async ({ page, sitePath }) => {
  await page.addInitScript(() => {
    Object.defineProperty(Storage.prototype, 'getItem', { value() { throw new Error('Storage unavailable'); } });
    Object.defineProperty(Storage.prototype, 'setItem', { value() { throw new Error('Storage unavailable'); } });
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async value => { window.copiedCommands = value; } } });
  });
  await page.goto(sitePath);
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('button', { name: 'Copy commands' }).click();
  await expect(page.getByRole('status')).toHaveText('Commands copied.');
  expect(await page.evaluate(() => window.copiedCommands)).toBe(await page.locator('#commands').textContent());
  await page.evaluate(() => { navigator.clipboard.writeText = async () => { throw new Error('Denied'); }; });
  await page.getByRole('button', { name: 'Copy commands' }).click();
  await expect(page.getByRole('status')).toHaveText('Clipboard unavailable. Select and copy the commands above.');
});
