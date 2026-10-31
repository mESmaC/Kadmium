const { test, expect } = require('@playwright/test');

for (const [shell, pages] of [
  ['/', ['overview', 'nodes', 'platforms', 'models', 'data', 'training', 'rl', 'gateway', 'slices', 'cloud', 'settings']],
  ['/companion', ['home', 'cluster', 'models', 'earn', 'tools', 'settings']]
]) {
  test(`${shell}: every workspace page fits desktop and mobile`, async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    for (const id of pages) {
      await page.goto(shell + (shell === '/' ? '#/' : '#') + id);
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
      await expect(page.locator('.cr-loading, .cp-loading')).toHaveCount(0);
      for (const width of [1180, 390]) {
        await page.setViewportSize({ width, height: 780 });
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${shell}${id} at ${width}px`).toBe(true);
      }
    }
    expect(errors).toEqual([]);
  });
}

for (const [name, path] of [['Control Room', '/'], ['Companion', '/companion']]) {
  test(`${name}: find features, navigate with keys, and restore focus`, async ({ page }) => {
    await page.goto(path);
    const trigger = page.getByRole('button', { name: 'Find a feature' });
    await trigger.click();
    const search = page.getByRole('searchbox', { name: 'Search features' });
    await expect(search).toBeFocused();
    await search.fill('mcp');
    await expect(page.locator('.kw-result')).toHaveCount(1);
    await expect(page.locator('.kw-result')).toContainText('API & playground');
    await search.press('ArrowDown');
    await expect(page.locator('.kw-result')).toBeFocused();
    await page.keyboard.press('ArrowUp');
    await expect(search).toBeFocused();
    await search.fill('no-such-feature');
    await expect(page.getByText('No matching features')).toBeVisible();
    await page.getByRole('button', { name: 'Clear search' }).click();
    await expect(search).toBeFocused();
    await expect(page.locator('.kw-result')).toHaveCount(14);
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toHaveCount(0);
    await expect(trigger).toBeFocused();
    await page.keyboard.press('Control+k');
    await expect(search).toBeFocused();
    // Native dialog keeps keyboard focus inside the feature finder.
    for (let i = 0; i < 18; i++) await page.keyboard.press('Tab');
    expect(await page.evaluate(() => !!document.activeElement.closest('dialog'))).toBe(true);
    await search.fill('models');
    await search.press('Enter');
    await expect(page).toHaveURL(name === 'Control Room' ? /#\/models$/ : /#models$/);
    await expect(page.getByRole('dialog')).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Models', exact: true, level: 1 })).toBeFocused();
  });

  test(`${name}: responsive surfaces and light theme remain usable`, async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(path);
    await expect(page.locator(name === 'Control Room' ? '.kw-journey' : '.cp-hero')).toBeVisible();
    for (const width of [1440, 1180, 940, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name} at ${width}px`).toBe(true);
      const trigger = page.getByRole('button', { name: 'Find a feature' });
      await expect(trigger).toBeVisible();
      await trigger.click();
      const dialog = page.getByRole('dialog');
      await expect(dialog).toBeVisible();
      const bounds = await dialog.boundingBox();
      expect(bounds.x).toBeGreaterThanOrEqual(0);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(width);
      await page.keyboard.press('Escape');
    }
    await page.setViewportSize({ width: 1180, height: 780 });
    if (name === 'Control Room') {
      await page.getByRole('button', { name: 'Switch to light theme' }).click();
    } else {
      await page.getByRole('button', { name: 'Settings', exact: true }).click();
      await page.getByRole('radio', { name: 'Light', exact: true }).click();
    }
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
    await page.getByRole('button', { name: 'Find a feature' }).click();
    await expect(page.getByRole('searchbox')).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test('Companion uses the desktop bridge for cluster tools and documentation', async ({ page }) => {
  await page.addInitScript(() => {
    window.desktopCalls = [];
    window.__KADMIUM_DESKTOP__ = {
      platform: 'windows', version: 'test',
      openControlRoom: hash => window.desktopCalls.push(['controlRoom', hash]),
      openExternal: url => window.desktopCalls.push(['external', url])
    };
  });
  await page.goto('/companion');
  await expect(page.locator('.kw-handoff')).toBeVisible();
  await page.getByRole('button', { name: 'Training (opens in browser)', exact: true }).click();
  await page.getByRole('button', { name: 'Find a feature' }).click();
  await page.getByRole('searchbox').fill('dream');
  await page.getByRole('searchbox').press('Enter');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.getByRole('button', { name: 'Find a feature' }).click();
  await page.getByRole('searchbox').fill('documentation');
  await page.getByRole('searchbox').press('Enter');
  expect(await page.evaluate(() => window.desktopCalls)).toEqual([
    ['controlRoom', '#/training'], ['controlRoom', '#/rl'], ['external', 'http://127.0.0.1:17948/docs']
  ]);
  await expect(page).toHaveURL(/\/companion$/);
});

test('Control Room setup guidance follows live workers and model state', async ({ page, request }) => {
  const overview = await (await request.get('/api/v1/cluster/overview')).json();
  const workers = await (await request.get('/api/v1/cluster/workers')).json();
  let live = 0, loaded = false;
  await page.route('**/api/v1/cluster/overview', route => route.fulfill({ json: { ...overview, active_model_loaded: loaded } }));
  await page.route('**/api/v1/cluster/workers', route => route.fulfill({ json: { ...workers, live_worker_count: live } }));
  for (const state of [[0, false, 'Connect a machine'], [0, true, 'Connect a machine'], [1, false, 'Choose a model'], [1, true, 'Open playground']]) {
    [live, loaded] = state;
    await page.goto('/');
    await expect(page.locator('.kw-journey').getByRole('button', { name: state[2], exact: true })).toBeVisible();
    await expect(page.locator('.kw-journey [aria-current="step"]')).toHaveCount(1);
    if (!live) await expect(page.getByRole('heading', { name: 'Your cluster, ready for work.' })).toHaveCount(0);
  }
  await page.locator('.kw-quick-actions summary').click();
  await expect(page.getByRole('link', { name: /Train a model/ })).toBeVisible();
});
