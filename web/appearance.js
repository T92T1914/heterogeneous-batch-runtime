/* Resolve the explicit preference before paint; Obscur is the default. */
(() => {
  'use strict';
  const key = 'heterogeneous-batch-runtime.appearance.v1';
  const root = document.documentElement;
  let choice = 'obscur';
  try {
    const saved = localStorage.getItem(key);
    if (['auto', 'clair', 'obscur'].includes(saved)) choice = saved;
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
        localStorage.setItem(key, choice);
      } catch { /* An in-memory choice remains usable. */ }
    });
  });
})();
