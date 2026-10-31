# Public website

Use this build for the public website at `kadmium.ai`, not for a hosted instance
of Control Room, Companion or Condenser. Keep managed cloud capabilities labelled
planned or simulated. Keep the maximum-quality contract separate from commercial
support proposals.

## Build

From the repository root, with Node 24 installed:

```sh
node scripts/build_website.mjs
```

Use `target/website` as the static publish directory and `/` as the site root.
To choose a different output directory:

```sh
node scripts/build_website.mjs --out-dir target/kadmium-ai
```

Publish the contents of that directory with your static host. Do not publish the
repository, state directory, gateway binary or workspace files. No deployment or
DNS change is performed by the build command. Configure the host for
`https://kadmium.ai/`; the page uses that canonical URL.

## Edit

Edit `crates/kadmium-gateway/src/marketing_site.html`, also used for the local
gateway’s `/site` preview. Do not fork the page into a separate hosted design.
Keep styling tied to `design-system/tokens.css`, its fonts and its logo assets.
The export includes the light and dark wordmarks for the theme control.

Documentation links point to repository documents rather than the local gateway’s
`/docs` or `/api/v1/docs` routes. Public calls to action should lead to setup
instructions or contact, not connect a visitor’s browser to local services.

Run the export again after changing the page or shared assets. Build into a
dedicated publish directory, not a directory containing other application output.

## Check

```sh
npm ci --prefix crates/kadmium-gateway/tests/ui
npm run test:website --prefix crates/kadmium-gateway/tests/ui
```

Install the Playwright Chromium browser if needed, or select installed Edge on
Windows before running the suite:

```powershell
$env:KADMIUM_TEST_BROWSER = 'msedge'
```

The website configuration serves the exported files on loopback for the tests.
Use the regular workspace suite to also check the embedded `/site` route:

```sh
npm test --prefix crates/kadmium-gateway/tests/ui -- --grep "marketing:"
```

Review the generated desktop dark/light and mobile screenshots in
`crates/kadmium-gateway/tests/ui/test-results/marketing-*.png`.
