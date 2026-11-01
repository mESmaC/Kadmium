import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const assets = [
  ['crates/kadmium-gateway/src/marketing_site.html', 'index.html'],
  ['design-system/tokens.css', 'tokens.css'],
  ['design-system/fonts/Archivo-Variable.woff2', 'fonts/Archivo-Variable.woff2'],
  ['design-system/fonts/JetBrainsMono-Variable.woff2', 'fonts/JetBrainsMono-Variable.woff2'],
  ['design-system/assets/Logos/kadmium-favicon.svg', 'assets/Logos/kadmium-favicon.svg'],
  ['design-system/assets/Logos/kadmium-wordmark-dark.svg', 'assets/Logos/kadmium-wordmark-dark.svg'],
  ['design-system/assets/Logos/kadmium-wordmark-light.svg', 'assets/Logos/kadmium-wordmark-light.svg']
];

export async function buildWebsite(outputDirectory = join(repositoryRoot, 'target/website')) {
  const destination = resolve(outputDirectory);
  for (const [source, target] of assets) {
    const outputPath = join(destination, target);
    await mkdir(dirname(outputPath), { recursive: true });
    await copyFile(join(repositoryRoot, source), outputPath);
  }
  return { directory: destination, files: assets.map(([, target]) => target) };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { values } = parseArgs({ options: { 'out-dir': { type: 'string' } } });
    const result = await buildWebsite(values['out-dir']);
    console.log('Website exported to ' + result.directory);
  } catch (error) {
    console.error('Website export failed: ' + error.message);
    process.exitCode = 1;
  }
}
