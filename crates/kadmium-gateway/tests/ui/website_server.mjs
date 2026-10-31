import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { buildWebsite } from '../../../../scripts/build_website.mjs';

const website = await buildWebsite();
const routes = new Map(website.files.map(file => ['/' + file, join(website.directory, file)]));
routes.set('/', join(website.directory, 'index.html'));
const contentTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.woff2', 'font/woff2']
]);
const server = createServer(async (request, response) => {
  const file = routes.get(request.url);
  if (!file || !['GET', 'HEAD'].includes(request.method)) {
    response.writeHead(404).end();
    return;
  }
  try {
    const bytes = await readFile(file);
    response.writeHead(200, { 'content-type': contentTypes.get(extname(file)) });
    response.end(request.method === 'HEAD' ? undefined : bytes);
  } catch {
    response.writeHead(500).end();
  }
});
server.requestTimeout = 5000;
server.listen(17948, '127.0.0.1');
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    server.close();
    server.closeAllConnections();
  });
}
