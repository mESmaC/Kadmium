const { defineConfig } = require('@playwright/test');
const path = require('node:path');

module.exports = defineConfig({
  testDir: '.',
  testMatch: 'marketing.spec.cjs',
  fullyParallel: true,
  workers: 2,
  use: {
    baseURL: 'http://127.0.0.1:17948',
    sitePath: '/',
    viewport: { width: 1440, height: 1000 },
    reducedMotion: 'reduce',
    channel: process.env.KADMIUM_TEST_BROWSER || undefined,
    trace: 'retain-on-failure'
  },
  webServer: {
    command: 'node website_server.mjs',
    cwd: path.resolve(__dirname),
    url: 'http://127.0.0.1:17948/',
    reuseExistingServer: false,
    timeout: 30000
  }
});
