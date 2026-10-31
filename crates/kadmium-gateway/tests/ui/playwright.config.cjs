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
    command: 'cargo run -p kadmium-gateway --example control_room_fixture -- --port 17948',
    cwd: path.resolve(__dirname, '../../../..'),
    env: {
      KADMIUM_STATE_DIR: path.resolve(__dirname, '../../../../target/workspace-ui-state'),
      KADMIUM_DISABLE_LIVE_PROBES: '1'
    },
    url: 'http://127.0.0.1:17948/workspace.js',
    reuseExistingServer: false,
    timeout: 300000
  }
});
