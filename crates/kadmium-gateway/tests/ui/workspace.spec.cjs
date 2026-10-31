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

test('Docs portal resolves catalog and repository-relative links', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/docs');
  const body = page.locator('#docBody');

  await body.getByRole('link', { name: 'Quickstart and installation', exact: true }).click();
  await expect(page.locator('#currentPath')).toHaveText('docs/getting-started.md');
  await body.getByRole('link', { name: 'launch guide', exact: true }).click();
  await expect(page.locator('#currentPath')).toHaveText('docs/guides/cli.md');

  await page.goto('/docs');
  await body.getByRole('link', { name: 'Measured benchmark ledger', exact: true }).click();
  await expect(page.locator('#currentPath')).toHaveText('docs/BENCHMARKS_2026-10.md');

  await page.goto('/docs');
  await body.getByRole('link', { name: 'USB-C delta bridges', exact: true }).click();
  await expect(page.locator('#currentPath')).toHaveText('docs/guides/usbc-delta-bridge.md');

  await page.goto('/docs');
  await body.getByRole('link', { name: 'Intel NPU kernels', exact: true }).click();
  await expect(page.locator('#currentPath')).toHaveText('crates/kadmium-npu/README.md');

  await page.goto('/docs');
  await body.getByRole('link', { name: 'Contributing', exact: true }).click();
  await expect(page.locator('#currentPath')).toHaveText('CONTRIBUTING.md');

  await page.evaluate(() => selectDoc('hardware-and-sharding'));
  await expect(body.getByRole('link', { name: 'Fts3ChunkSlicer', exact: true })).toHaveAttribute(
    'href',
    'https://github.com/mESmaC/Kadmium/blob/main/crates/kadmium-data/src/lib.rs'
  );
  expect(errors).toEqual([]);
});

test('Docs portal section links scroll, focus, reload, and navigate back', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const filler = '\n\nParagraph before the target.\n'.repeat(70);
  await page.route('**/api/v1/docs', route => route.fulfill({ json: [
    {
      slug: 'overview', file_path: 'docs/README.md', title: 'Overview', category: 'Docs', summary: '',
      markdown: '# Overview\n[Local target](#-local-section)\n' + filler +
        '## 🧪 **Local** `section`\n[Contributor target](../CONTRIBUTING.md#caf%C3%A9-checks)'
    },
    {
      slug: 'contributing', file_path: 'CONTRIBUTING.md', title: 'Contributing', category: 'Docs', summary: '',
      markdown: '# Contributing\n' + filler + '## Café checks\n\nReady.'
    }
  ] }));
  await page.goto('/docs');
  await page.locator('#docBody').getByRole('link', { name: 'Local target' }).click();
  const local = page.getByRole('heading', { name: '🧪 Local section', exact: true });
  await expect(page).toHaveURL(/#overview\/-local-section$/);
  await expect(local).toHaveAttribute('id', '-local-section');
  await expect(local).toBeFocused();
  await expect(local).toBeInViewport();
  expect((await local.boundingBox()).y).toBeGreaterThanOrEqual(60);

  await page.locator('#docBody').getByRole('link', { name: 'Contributor target' }).click();
  await expect(page).toHaveURL(/#contributing\/caf%C3%A9-checks$/);
  await expect(page.locator('#currentPath')).toHaveText('CONTRIBUTING.md');
  await expect(page.getByRole('heading', { name: 'Café checks' })).toBeFocused();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Café checks' })).toBeFocused();
  await expect(page.getByRole('heading', { name: 'Café checks' })).toBeInViewport();
  await page.goBack();
  await expect(local).toBeFocused();
  await expect(local).toBeInViewport();
  expect(errors).toEqual([]);
});

test('Docs portal gives duplicate headings unique IDs and tolerates malformed fragments', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/api/v1/docs', route => route.fulfill({ json: [{
    slug: 'overview', file_path: 'docs/README.md', title: 'Overview', category: 'Docs', summary: '',
    markdown: '# Overview\n[Second heading](#repeat-1)\n## Repeat\n## Repeat\n' +
      '##### Repeat-1\n###### [Linked](https://example.com) heading ###'
  }] }));
  await page.goto('/docs#overview/%invalid');
  await expect(page.locator('#docBody h1')).toHaveText('Overview');
  expect(await page.locator('#docBody :is(h1,h2,h5,h6)').evaluateAll(nodes => nodes.map(n => n.id)))
    .toEqual(['overview', 'repeat', 'repeat-1', 'repeat-1-1', 'linked-heading']);
  await page.locator('#docBody').getByRole('link', { name: 'Second heading' }).click();
  await expect(page.locator('#repeat-1')).toBeFocused();
  expect(errors).toEqual([]);
});
