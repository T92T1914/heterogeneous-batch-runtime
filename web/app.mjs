const metricLabels = {
  median_ms: 'Complete host call', kernel_ms: 'CUDA kernel interval',
  upload_ms: 'CUDA upload interval', download_ms: 'CUDA download interval',
};

export function selectedRows(study, condition) {
  return study.rows.filter(row => row.condition === condition);
}

export function timingView(study, condition, metric) {
  if (!Object.hasOwn(metricLabels, metric)) throw new Error('Unknown timing boundary');
  const rows = selectedRows(study, condition);
  if (!rows.length) throw new Error('Unknown condition');
  const values = rows.map(row => row[metric]).filter(value => value !== null);
  if (values.some(value => !Number.isFinite(value) || value < 0)) throw new Error('Invalid timing interval');
  return {rows, maximum: values.length ? Math.max(...values) : 0, label: metricLabels[metric]};
}

export function parseSelection(data, hash) {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const requested = data.experiments.find(study => study.id === params.get('experiment'));
  const study = requested ?? data.experiments.find(item => item.id === data.default_selection.experiment);
  const wanted = requested ? params.get('condition') : data.default_selection.condition;
  const condition = study.conditions.find(item => item.id === wanted)?.id ?? study.conditions[0].id;
  let metric = Object.hasOwn(metricLabels, params.get('metric')) ? params.get('metric') : 'median_ms';
  if (!study.rows.some(row => row[metric] !== null)) metric = 'median_ms';
  return {study, condition, metric};
}

function element(name, text, attributes = {}) {
  const node = document.createElement(name);
  if (text !== undefined) node.textContent = text;
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function link(text, url) { return element('a', text, {href:url}); }
function formatted(value) { return value === null ? 'Not measured' : `${value.toFixed(5)} ms`; }

function renderTable(rows, metric, label) {
  const region = element('div', undefined, {class:'table-wrap', role:'region', tabindex:'0', 'aria-label':'Selected comparison values'});
  const table = element('table');
  table.append(element('caption', `${label}. Medians in milliseconds, shown to five decimal places.`));
  const headers = metric === 'median_ms' ? ['Path','Warm samples','Median ms','Middle 50% ms','Minimum ms','Maximum ms','First ms'] : ['Path','Warm samples',`${label} median ms`];
  const head = element('thead'), headRow = element('tr');
  headers.forEach(text => headRow.append(element('th', text, {scope:'col'})));
  head.append(headRow);
  const body = element('tbody');
  rows.forEach(row => {
    const tr = element('tr');
    tr.dataset.method = row.method;
    tr.append(element('th', row.method_label, {scope:'row'}));
    const numbers = metric === 'median_ms' ? [row.samples, row.median_ms.toFixed(5), `${row.p25_ms.toFixed(5)} to ${row.p75_ms.toFixed(5)}`, row.min_ms.toFixed(5), row.max_ms.toFixed(5), row.first_ms.toFixed(5)] : [row.samples, row[metric] === null ? 'Not measured' : row[metric].toFixed(5)];
    numbers.forEach(text => tr.append(element('td', String(text))));
    body.append(tr);
  });
  table.append(head, body);
  region.append(table);
  return region;
}

function boot() {
  try {
    const data = JSON.parse(document.getElementById('evidence-data').textContent);
    const controls = {experiment:document.getElementById('experiment'), condition:document.getElementById('condition'), metric:document.getElementById('metric')};
    let selection = parseSelection(data, location.hash);
    data.experiments.forEach(study => controls.experiment.append(element('option', study.title, {value:study.id})));

    function render(updateURL = false) {
      const {study, condition, metric} = selection;
      controls.experiment.value = study.id;
      controls.condition.replaceChildren(...study.conditions.map(item => element('option', item.label, {value:item.id})));
      controls.condition.value = condition;
      for (const option of controls.metric.options) option.disabled = !study.rows.some(row => row[option.value] !== null);
      controls.metric.value = metric;
      const view = timingView(study, condition, metric);
      const conditionLabel = study.conditions.find(item => item.id === condition).label;
      document.getElementById('comparison-title').textContent = study.title;
      document.getElementById('description').textContent = study.description;
      document.getElementById('selection-status').textContent = `${conditionLabel}. ${view.label}. ${view.rows[0].samples} warm samples per path. ${metric === 'median_ms' ? 'Setup and first invocations are recorded separately where applicable.' : 'CPU device intervals are not measured. These values do not represent complete-call time.'}`;
      document.getElementById('chart-heading').textContent = `${conditionLabel}: ${view.label.toLowerCase()}`;
      document.getElementById('chart-maximum').textContent = `${view.maximum.toFixed(5)} ms`;
      const chart = document.getElementById('chart');
      chart.replaceChildren();
      view.rows.forEach(row => {
        const item = element('div', undefined, {class:'bar-row'});
        item.dataset.method = row.method;
        item.dataset.kind = study.id === 'inference-cpu' || row.method.startsWith('cpu') || row.method.includes('python-cpu') || row.method.startsWith('scalar') || row.method.startsWith('optimized') || row.method.startsWith('original') || row.method.startsWith('followup') ? 'cpu' : row.method.includes('reuse') || row.method === 'cpp-reused' ? 'reused' : 'cuda';
        item.append(element('span', row.method_label), element('span', formatted(row[metric]), {class:'bar-value'}));
        const track = element('div', undefined, {class:'bar-track', 'aria-hidden':'true'});
        if (row[metric] === null) track.append(element('span', 'Not measured', {class:'missing'}));
        else {
          const bar = element('div', undefined, {class:'bar'});
          bar.style.width = `${view.maximum === 0 ? 0 : row[metric] / view.maximum * 100}%`;
          track.append(bar);
        }
        item.append(track);
        chart.append(item);
      });
      document.getElementById('selection-table').replaceChildren(renderTable(view.rows, metric, view.label));
      document.getElementById('finding').textContent = study.finding;
      for (const key of ['boundary', 'excluded', 'limitations']) document.getElementById(key).textContent = study[key];
      const sources = document.getElementById('source-links');
      sources.replaceChildren(document.createTextNode('Executed source '), link(study.source_revision, study.source_url), document.createTextNode('. '), link('Raw record', study.raw_url), document.createTextNode(', '), link('protocol at executed source', study.protocol_url), document.createTextNode(', '), link('report', study.report_url), document.createTextNode('.'));
      if (study.comparative_source_revision) sources.append(document.createTextNode(' Original source '), link(study.comparative_source_revision, study.comparative_source_url), document.createTextNode(' and '), link('original CPU rows', study.secondary_raw_url), document.createTextNode(' remain separate.'));
      document.getElementById('table-link').href = `#table-${study.id}`;
      if (updateURL) {
        const params = new URLSearchParams({experiment:study.id, condition, metric});
        try { history.replaceState(null, '', `#${params}`); } catch { /* Reading does not depend on URL updates. */ }
      }
    }

    controls.experiment.addEventListener('change', () => {
      const study = data.experiments.find(item => item.id === controls.experiment.value);
      selection = {study, condition:study.conditions[0].id, metric:'median_ms'};
      render(true);
    });
    controls.condition.addEventListener('change', () => { selection.condition = controls.condition.value; render(true); });
    controls.metric.addEventListener('change', () => { selection.metric = controls.metric.value; render(true); });
    window.addEventListener('hashchange', () => {
      if (!location.hash.startsWith('#experiment=')) return;
      selection = parseSelection(data, location.hash);
      render();
    });
    const openTarget = () => {
      const id = location.hash.slice(1);
      if (!/^table-[a-z-]+$/.test(id)) return;
      const target = document.getElementById(id);
      if (target instanceof HTMLDetailsElement) target.open = true;
    };
    window.addEventListener('hashchange', openTarget);
    openTarget();
    document.querySelectorAll('a[href^="#table-"],#table-link').forEach(anchor => anchor.addEventListener('click', () => {
      const target = document.getElementById(anchor.hash.slice(1));
      if (target instanceof HTMLDetailsElement) target.open = true;
    }));
    render();
    document.getElementById('interactive').hidden = false;
  } catch {
    document.getElementById('interactive').hidden = true;
    document.getElementById('explorer-error').hidden = false;
  }
}

if (typeof document !== 'undefined') boot();
