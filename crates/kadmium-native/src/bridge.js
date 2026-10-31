(function () {
  'use strict';
  // Initialization scripts also run in subframes. Only the application page gets a bridge.
  if (window !== window.top || location.origin !== __KADMIUM_ORIGIN__) return;
  let pending = null;
  window.__KADMIUM_FILES__ = Object.freeze({
    pick(options) {
      if (pending) return Promise.reject(new Error('A file dialog is already open.'));
      return new Promise((resolve, reject) => {
        const id = crypto.randomUUID();
        pending = { id, resolve, reject };
        try {
          const body = JSON.stringify({ ...options, cmd: 'pickPath', id });
          if (new TextEncoder().encode(body).length > 32 * 1024) {
            throw new Error('The path is too long for the native dialog. Enter a shorter path or use the embedded browser.');
          }
          window.ipc.postMessage(body);
        } catch (error) {
          pending = null;
          reject(error);
        }
      });
    },
    complete(result) {
      // A response from an old page must never reach a new page after navigation.
      if (!pending || result.id !== pending.id) return;
      const request = pending;
      pending = null;
      if (result.error) request.reject(new Error(result.error));
      else request.resolve(result.path);
    }
  });
})();
