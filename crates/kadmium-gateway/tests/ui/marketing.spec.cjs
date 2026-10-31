const { test, expect } = require('@playwright/test');

// Exercise the public page through the real gateway, including its embedded assets.
test('marketing: service boundaries and local setup remain readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  try {
    await page.goto('http://127.0.0.1:17948/site');
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Your machines.');
    await expect(page.locator('#services')).toContainText('Paid prices are proposals for discussion');
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

test('marketing: responsive page, assets, anchors and documentation destinations', async ({ page, request }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()}: ${response.url()}`); });
  await page.goto('/site');
  await page.evaluate(() => document.fonts.ready);
  for (const width of [1440, 1180, 940, 768, 760, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow at ${width}px`).toBe(true);
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  }
  const missingAnchors = await page.locator('a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)).filter(id => !document.getElementById(id)));
  expect(missingAnchors).toEqual([]);
  const docs = await (await request.get('/api/v1/docs')).json();
  const docLinks = await page.locator('a[href^="/docs#"]').evaluateAll(links => links.map(link => link.hash.slice(1)));
  for (const slug of docLinks) expect(docs.some(doc => doc.slug === slug), slug).toBe(true);
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

test('marketing: keyboard navigation, mobile menu and resize recovery', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/site');
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

test('marketing: copy commands and recover from denied clipboard or storage', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(Storage.prototype, 'getItem', { value() { throw new Error('Storage unavailable'); } });
    Object.defineProperty(Storage.prototype, 'setItem', { value() { throw new Error('Storage unavailable'); } });
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async value => { window.copiedCommands = value; } } });
  });
  await page.goto('/site');
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('button', { name: 'Copy commands' }).click();
  await expect(page.getByRole('status')).toHaveText('Commands copied.');
  expect(await page.evaluate(() => window.copiedCommands)).toBe(await page.locator('#commands').textContent());
  await page.evaluate(() => { navigator.clipboard.writeText = async () => { throw new Error('Denied'); }; });
  await page.getByRole('button', { name: 'Copy commands' }).click();
  await expect(page.getByRole('status')).toHaveText('Clipboard unavailable. Select and copy the commands above.');
});
