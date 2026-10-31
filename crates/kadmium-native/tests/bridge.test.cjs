const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../src/bridge.js'), 'utf8')
  .replace('__KADMIUM_ORIGIN__', JSON.stringify('http://127.0.0.1:7856'));

function bridge({ frame = false, origin = 'http://127.0.0.1:7856', fail = false } = {}) {
  const messages = [];
  let id = 0;
  const window = { ipc: { postMessage(text) { if (fail) throw Error('IPC unavailable'); messages.push(JSON.parse(text)); } } };
  window.top = frame ? {} : window;
  vm.runInNewContext(script, { window, location: { origin }, TextEncoder, crypto: { randomUUID: () => 'request-' + ++id } });
  return { picker: window.__KADMIUM_FILES__, messages };
}

test('native bridge is absent in frames and other origins', () => {
  assert.equal(bridge({ frame: true }).picker, undefined);
  assert.equal(bridge({ origin: 'https://example.com' }).picker, undefined);
});

test('selection and cancellation settle once; duplicate dialogs are refused', async () => {
  const { picker, messages } = bridge();
  const first = picker.pick({ kind: 'model_file' });
  await assert.rejects(picker.pick({ kind: 'model_folder' }), /already open/);
  assert.equal(messages.length, 1);
  picker.complete({ id: 'stale-page-id', path: 'wrong.gguf' });
  picker.complete({ id: messages[0].id, path: 'C:\\models\\quoted"name.gguf' });
  assert.equal(await first, 'C:\\models\\quoted"name.gguf');
  const cancel = picker.pick({ kind: 'model_folder' });
  picker.complete({ id: messages[1].id, path: null });
  assert.equal(await cancel, null);
  const error = picker.pick({ kind: 'workspace_file', root: 'workspace' });
  picker.complete({ id: messages[2].id, error: 'Select a file inside the active workspace.' });
  await assert.rejects(error, /inside the active workspace/);
});

test('IPC failures clear the pending request for retry', async () => {
  const { picker } = bridge({ fail: true });
  await assert.rejects(picker.pick({ kind: 'model_file' }), /IPC unavailable/);
  await assert.rejects(picker.pick({ kind: 'model_file' }), /IPC unavailable/);
});

test('a native validation rejection releases the chooser for a corrected request', async () => {
  const { picker, messages } = bridge();
  const invalid = picker.pick({ kind: 'workspace_file' });
  picker.complete({ id: messages[0].id, error: 'The workspace is not ready. Refresh and try again.' });
  await assert.rejects(invalid, /workspace is not ready/);
  const retry = picker.pick({ kind: 'workspace_file', root: 'workspace' });
  assert.equal(messages.length, 2);
  picker.complete({ id: messages[1].id, path: 'hello.txt' });
  assert.equal(await retry, 'hello.txt');
});

test('oversized UTF-8 paths fail visibly without leaving a pending dialog', async () => {
  const { picker, messages } = bridge();
  await assert.rejects(picker.pick({ kind: 'model_file', path: '界'.repeat(12000) }), /too long/);
  assert.equal(messages.length, 0);
  const retry = picker.pick({ kind: 'model_folder' });
  picker.complete({ id: messages[0].id, path: null });
  assert.equal(await retry, null);
});
