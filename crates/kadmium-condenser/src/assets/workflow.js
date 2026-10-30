(function (root) {
  'use strict';

  function credentialKey(kind, baseUrl) {
    return kind + ':' + String(baseUrl || '').trim().replace(/\/+$/, '');
  }

  function forkTask(source, id, now) {
    const task = JSON.parse(JSON.stringify(source));
    Object.assign(task, {
      id, parentId: source.id, title: source.title + ' (fork)',
      createdAt: now, updatedAt: now, status: 'ready', lastError: null,
      queue: [], queuePaused: true
    });
    return task;
  }

  function branchTask(source, messageIndex, id, now) {
    const message = source.messages[messageIndex];
    if (!message || message.role !== 'user') throw new Error('Choose a user prompt to branch from.');
    const task = forkTask(source, id, now);
    task.branchPoint = messageIndex;
    // Cut only at user boundaries so assistant tool calls and their results stay paired.
    task.messages = task.messages.slice(0, messageIndex);
    task.runs = task.runs.filter(run => Number.isInteger(run.message_end) && run.message_end <= messageIndex);
    task.draft = message.content || '';
    task.attachments = [];
    task.config.interview = false;
    return task;
  }

  function queuedPrompt(prompt, attachments, config, id, now) {
    const item = JSON.parse(JSON.stringify({ id, prompt: prompt.trim(), attachments, config, createdAt: now }));
    delete item.config.api_key;
    item.config.interview = false;
    if (!item.prompt && !item.attachments.length) throw new Error('Enter a follow-up or attach a file.');
    return item;
  }

  function canContinueQueue(session) {
    return session.status === 'complete' && !session.queuePaused && session.queue.length > 0;
  }

  // Decode complete SSE frames, including UTF-8 and CRLF split across reads.
  // A missing final report is an interrupted run, never a reason to submit it again.
  async function readRunStream(response, onEvent) {
    if (!(response.headers.get('content-type') || '').toLowerCase().startsWith('text/event-stream') || !response.body) {
      throw new Error('The server did not return a run event stream.');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let data = [];
    let report = null;
    const dispatch = () => {
      if (!data.length) return;
      const event = JSON.parse(data.join('\n'));
      data = [];
      if (event.event === 'error') throw new Error(event.error || 'Run failed');
      if (event.event === 'finished') report = event.report;
      onEvent(event);
    };
    const line = (value) => {
      value = value.replace(/\r$/, '');
      if (!value) dispatch();
      else if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /, ''));
    };
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf('\n')) !== -1) {
          line(buffer.slice(0, index));
          buffer = buffer.slice(index + 1);
        }
        if (done) break;
      }
      if (buffer) line(buffer);
      dispatch();
      if (!report || !Array.isArray(report.messages)) {
        throw new Error('Connection ended before the final run report. Partial output is preserved; the request was not retried.');
      }
      return report;
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }

  const workflow = { credentialKey, forkTask, branchTask, queuedPrompt, canContinueQueue, readRunStream };
  if (typeof module === 'object' && module.exports) module.exports = workflow;
  else root.CondenserWorkflow = workflow;
})(globalThis);
