# Workspace browser checks

From this directory, run `npm ci`, `npx playwright install chromium`, and `npm test`.
Playwright starts the real Rust gateway on port 17948 (worker port 17949), or reuses
an existing local test gateway. No model loading, training, or cloud mutations are
performed. Setup-state coverage intercepts read-only responses in the browser.

To use an installed Microsoft Edge browser on Windows:
`$env:KADMIUM_TEST_BROWSER='msedge'; npm test`.

Coverage includes both shells at 320–1440px, dark and light themes, feature search,
keyboard navigation, dialog focus containment/restoration, native desktop handoff,
and setup guidance. Traces are retained only on failure.

The public marketing page at `/site` is covered for service availability boundaries,
JavaScript-disabled reading, local setup and documentation links, responsive layout,
mobile navigation, keyboard focus, theme persistence and clipboard failure recovery.
Page screenshots are written to `test-results/marketing-*.png`. Run just these checks
with `npm test -- --grep 'marketing:'`.

For the standalone kadmium.ai export, run `npm run test:website`. This runs the
same marketing checks against static files without starting the Rust gateway.
Coverage also checks the product/quality/cloud boundaries, public documentation
destinations, shared-asset copies, export errors and the absence of local API
requests. See [Public website](../../../../docs/website.md) for build and hosting
instructions.
