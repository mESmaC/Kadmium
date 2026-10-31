const { defineConfig } = require('@playwright/test');
const path = require('node:path');

module.exports = defineConfig({
  testDir: '.',
  testMatch: '*.spec.cjs',
  fullyParallel: true,
  workers: 2,
  use: {
    baseURL: 'http://127.0.0.1:17948',
    viewport: { width: 1440, height: 1000 },
    reducedMotion: 'reduce',
    channel: process.env.KADMIUM_TEST_BROWSER || undefined,
    trace: 'retain-on-failure'
  },
  webServer: {
    command: 'cargo run -p kadmium-cli --bin kadmium -- serve --port 17948 --worker-port 17949',
    cwd: path.resolve(__dirname, '../../../..'),
    url: 'http://127.0.0.1:17948/workspace.js',
    reuseExistingServer: !process.env.CI,
    timeout: 300000
  }
});
