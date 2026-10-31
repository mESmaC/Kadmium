const { test } = require('node:test');
const assert = require('node:assert/strict');
const { credentialKey, forkTask, branchTask, queuedPrompt, canContinueQueue, readRunStream } = require('../src/assets/workflow.js');

function stream(text, chunkSize = 1) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize));
      controller.close();
    }
  }), { headers: { 'content-type': 'text/event-stream' } });
}

test('stream decoder preserves split UTF-8, CRLF, multiline events and final frames', async () => {
  const events = [];
  const report = { status: 'completed', messages: [{ role: 'assistant', content: 'Hello 🌍' }] };
  const response = stream(': keepalive\r\ndata: {"event":"model_token",\r\ndata: "delta":"🌍"}\r\n\r\ndata: ' + JSON.stringify({ event: 'finished', report }));
  assert.deepEqual(await readRunStream(response, event => events.push(event)), report);
  assert.equal(events[0].delta, '🌍');
  assert.equal(events.length, 2);
});

test('a truncated or failed run preserves observed events and fails without replay', async () => {
  const events = [];
  await assert.rejects(readRunStream(stream('data: {"event":"model_token","delta":"partial"}\n\n'), e => events.push(e)), /not retried/);
  assert.equal(events[0].delta, 'partial');
  await assert.rejects(readRunStream(stream('data: {"event":"error","error":"provider unavailable"}\n\n'), () => {}), /provider unavailable/);
  await assert.rejects(readRunStream(stream('data: invalid\n\n'), () => {}), SyntaxError);
});

test('forks preserve conversation, drafts and run routes with independent nested state', () => {
  const source = { id: 'source', title: 'Task', status: 'complete', messages: [{ content: 'Hello' }], draft: 'Next question', attachments: [{ content: 'context' }], config: { kind: 'kadmium', sampling: { temperature: 0.2 } }, runs: [{ route: { model: 'model-a' } }] };
  const fork = forkTask(source, 'fork', 123);
  assert.equal(fork.parentId, source.id);
  assert.equal(fork.draft, source.draft);
  assert.deepEqual(fork.runs, source.runs);
  fork.config.sampling.temperature = 0.9;
  fork.messages[0].content = 'Changed';
  fork.attachments[0].content = 'Changed';
  assert.equal(source.config.sampling.temperature, 0.2);
  assert.equal(source.messages[0].content, 'Hello');
  assert.equal(source.attachments[0].content, 'context');
});

test('credentials are scoped to both provider and endpoint', () => {
  assert.equal(credentialKey('openai', 'http://localhost:8080/'), credentialKey('openai', 'http://localhost:8080'));
  assert.notEqual(credentialKey('openai', 'http://localhost:8080'), credentialKey('openai', 'https://api.openai.com'));
  assert.notEqual(credentialKey('kadmium', 'http://localhost'), credentialKey('openai', 'http://localhost'));
});

test('branching at a user turn preserves tool pairs and drops future runs and queued actions', () => {
  const source = { id: 'source', title: 'Task', config: { interview: true }, messages: [
    { role: 'user', content: 'First' },
    { role: 'assistant', tool_calls: [{ id: 'call' }] },
    { role: 'tool', tool_call_id: 'call', content: 'Result' },
    { role: 'assistant', content: 'Done' },
    { role: 'user', content: 'Revise me' },
    { role: 'assistant', content: 'Future answer' }
  ], runs: [{ message_end: 4 }, { message_end: 6 }, { status: 'legacy' }], queue: [{ prompt: 'Do something' }], attachments: [{ name: 'draft.txt' }] };
  const branch = branchTask(source, 4, 'branch', 123);
  assert.deepEqual(branch.messages, source.messages.slice(0, 4));
  assert.deepEqual(branch.runs, [{ message_end: 4 }]);
  assert.equal(branch.draft, 'Revise me');
  assert.equal(branch.config.interview, false);
  assert.equal(branch.branchPoint, 4);
  assert.deepEqual(branch.queue, []);
  assert.deepEqual(branch.attachments, []);
  branch.messages[2].content = 'Changed';
  assert.equal(source.messages[2].content, 'Result');
  assert.equal(source.queue.length, 1);
  assert.deepEqual(forkTask(source, 'fork', 123).queue, []);
  assert.throws(() => branchTask(source, 2, 'invalid', 123), /user prompt/);
});

test('queued prompts freeze routing, permissions, instructions and attachments without credentials', () => {
  const config = { kind: 'openai', model: 'model-a', base_url: 'http://localhost', api_key: 'secret', allowed_tools: ['files__read'], system_prompt: 'Be concise', sampling: { temperature: 0.2 }, interview: true, memory_enabled: true, research_required: true };
  const attachments = [{ name: 'context.txt', content: 'Snapshot' }];
  const item = queuedPrompt(' Follow up ', attachments, config, 'queued', 123);
  config.model = 'model-b'; config.allowed_tools.push('files__write'); attachments[0].content = 'Changed';
  config.memory_enabled = false; config.research_required = false;
  assert.equal(item.config.memory_enabled, true);
  assert.equal(item.config.research_required, true);
  assert.equal(item.config.model, 'model-a');
  assert.deepEqual(item.config.allowed_tools, ['files__read']);
  assert.equal(item.config.system_prompt, 'Be concise');
  assert.equal(item.attachments[0].content, 'Snapshot');
  assert.equal(item.prompt, 'Follow up');
  assert.equal(item.config.interview, false);
  assert.equal(item.config.api_key, undefined);
  assert.throws(() => queuedPrompt(' ', [], config, 'empty', 123), /follow-up/);
});

test('queues advance only after success and never while paused or empty', () => {
  for (const status of ['running', 'ready', 'error', 'cancelled']) {
    assert.equal(canContinueQueue({ status, queuePaused: false, queue: [{}] }), false);
  }
  assert.equal(canContinueQueue({ status: 'complete', queuePaused: false, queue: [{}] }), true);
  assert.equal(canContinueQueue({ status: 'complete', queuePaused: true, queue: [{}] }), false);
  assert.equal(canContinueQueue({ status: 'complete', queuePaused: false, queue: [] }), false);
});
