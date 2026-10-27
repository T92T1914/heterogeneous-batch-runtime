import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import vm from 'node:vm';
import {parseSelection, timingView} from '../web/app.mjs';

const data = JSON.parse(await readFile(new URL('../web/evidence.json', import.meta.url)));
const appearance = await readFile(new URL('../web/appearance.js', import.meta.url), 'utf8');

function preference(saved, blocked = false) {
  const state = {saved}, events = new Map(), root = {dataset:{}};
  const control = {addEventListener:(name, callback)=>events.set(name, callback)};
  vm.runInNewContext(appearance, {
    document:{documentElement:root, getElementById:()=>control,
      addEventListener:(name, callback)=>events.set(name, callback)},
    localStorage:{getItem:()=>{if(blocked)throw Error('blocked');return state.saved;},
      setItem:(_, value)=>{if(blocked)throw Error('blocked');state.saved=value;}}
  });
  return {root, state, choose:value=>{events.get('DOMContentLoaded')();control.value=value;events.get('change')();}};
}

test('Obscur default and saved explicit modes are resolved before paint', () => {
  for (const saved of [null, 'invalid']) assert.equal(preference(saved).root.dataset.appearance, 'obscur');
  for (const saved of ['auto', 'clair', 'obscur']) assert.equal(preference(saved).root.dataset.appearance, saved);
});

test('Auto survives a fresh document and blocked storage keeps in-page switching', () => {
  const first = preference('clair');first.choose('auto');
  assert.equal(first.state.saved, 'auto');
  assert.equal(preference(first.state.saved).root.dataset.appearance, 'auto');
  const blocked = preference('clair', true);
  assert.equal(blocked.root.dataset.appearance, 'obscur');blocked.choose('clair');
  assert.equal(blocked.root.dataset.appearance, 'clair');
});

test('invalid and absent URL choices resolve to a real retained default', () => {
  for (const hash of ['', '#experiment=missing&condition=missing&metric=missing', '#contracts']) {
    const selection = parseSelection(data, hash);
    assert.equal(selection.study.id, 'reuse');
    assert.equal(selection.condition, '1024:uniform');
    assert.equal(selection.metric, 'median_ms');
  }
});

test('requested inference cannot acquire a fictional device timing lens', () => {
  const selection = parseSelection(data, '#experiment=inference-cuda&condition=8&metric=kernel_ms');
  assert.equal(selection.condition, '8');
  assert.equal(selection.metric, 'median_ms');
  const view = timingView(selection.study, '8', 'median_ms');
  assert.equal(view.rows.length, 3);
  assert.ok(view.rows.find(row => row.method === 'python-cpu-reused').median_ms < view.rows.find(row => row.method === 'cpp-reused').median_ms);
});

test('device-event selection keeps unmeasured CPU paths and the separate scale', () => {
  const study = data.experiments.find(item => item.id === 'reuse');
  const view = timingView(study, '1024:uniform', 'kernel_ms');
  assert.equal(view.rows.length, 6);
  assert.equal(view.rows[0].kernel_ms, null);
  assert.equal(view.rows[1].kernel_ms, null);
  assert.ok(view.maximum < timingView(study, '1024:uniform', 'median_ms').maximum);
  assert.equal(view.rows.find(row => row.method === 'cuda_reuse_shared').method_label, 'Reused CUDA, shared bins');
});

test('unknown condition and metric are rejected rather than charted', () => {
  const study = data.experiments[0];
  assert.throws(() => timingView(study, 'missing', 'median_ms'), /Unknown condition/);
  assert.throws(() => timingView(study, study.conditions[0].id, 'missing'), /Unknown timing boundary/);
});
