/* Apply a project-scoped preference before paint. Auto also works without JS. */
(() => {
  'use strict';
  const key = 'heterogeneous-batch-runtime.appearance.v1';
  const root = document.documentElement;
  let choice = 'auto';
  try {
    const saved = localStorage.getItem(key);
    if (saved === 'clair' || saved === 'obscur') choice = saved;
  } catch { /* Optional storage must not prevent reading. */ }
  root.dataset.appearance = choice;
  document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('appearance');
    if (!select) return;
    select.value = choice;
    select.disabled = false;
    select.addEventListener('change', () => {
      choice = ['clair', 'obscur'].includes(select.value) ? select.value : 'auto';
      root.dataset.appearance = choice;
      try {
        if (choice === 'auto') localStorage.removeItem(key);
        else localStorage.setItem(key, choice);
      } catch { /* An in-memory choice remains usable. */ }
    });
  });
})();
