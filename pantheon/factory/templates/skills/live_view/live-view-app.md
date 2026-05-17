---
id: live_view_app
name: Generate an Agent-Controllable LiveView App
description: |
  Write a custom interactive visualization component that the agent can
  open, drive, and observe — using the LiveView SDK. Covers the SDK API,
  the component contract, how to serve and open it, and how the agent
  drives it via live_view_update / live_view_call.
tags: [live-view, sdk, custom-component, visualization, interactive, generate-app]
---

# Generating an Agent-Controllable LiveView App

When no existing viewer fits, write your own interactive component. With the
**LiveView SDK** the component is controllable-by-construction: the agent
drives it and reads it back, with no extra wiring.

Use this for bespoke dashboards, custom plots, tailored data views. For
heavy domain viewers (spatial omics) prefer the Vitessce skill instead.

## The component contract

A component is a **JS ES module** that exports `setup(lv, root)`:

```js
// my-app.js
export function setup(lv, root) {
  // `root` : the container element to render into
  // `lv`   : the LiveView SDK instance

  lv.onState((state) => {
    // Called on the initial state and after every agent update.
    // `state` is the full current state. Render it into `root`.
  })

  lv.defineAction('do_something', async (args) => {
    // A command the agent can invoke via live_view_call.
    // The return value is sent back to the agent.
    return { ok: true }
  })

  // When the USER interacts, report the new state so the agent sees it:
  //   lv.emitState(newState)
}
```

`setup` may be async. The host calls `lv.ready()` once it resolves.

## Two component styles

**Vanilla** (above) — a `.js` module exporting `setup(lv, root)`; you render
into `root` yourself (innerHTML, DOM, SVG, canvas, any library).

**React / JSX** — write a `.jsx` (or `.tsx`) module with a **default-exported
React component**; the host transpiles it in-browser (Sucrase) and mounts it,
passing the SDK instance as the `lv` prop:

```jsx
// my-app.jsx  — JSX works directly, no need to import React
import { useState, useEffect } from 'react'

export default function App({ lv }) {
  const [state, setState] = useState(lv.state)
  useEffect(() => lv.onState(setState), [])      // agent updates → re-render
  if (!state) return <div>Loading…</div>
  return (
    <button onClick={() => lv.emitState({ ...state, n: (state.n || 0) + 1 })}>
      clicked {state.n || 0}
    </button>
  )
}
```

Use whichever fits. React/JSX suits richer UIs; vanilla suits small or
library-driven views. Both get the same `lv` — same control loop.

## SDK API (live-view-sdk.js)

| Call | Purpose |
|------|---------|
| `lv.onState((state, info) => …)` | Render callback. Fires on init + every update with the full merged state. `info.reason` ∈ `init`/`patch`/`set`. |
| `lv.defineAction(name, fn)` | Expose a command. `fn(args)` (sync/async); its return value goes back to the agent. |
| `lv.emitState(state)` | Report component state outward after a user interaction (debounced). |
| `lv.state` | The current merged state. |
| `lv.fail(msg)` | Report an unrecoverable error. |

The module is an ES module — it may `import` libraries from a CDN, e.g.
`import Plotly from 'https://esm.sh/plotly.js-dist-min'`.

## Workflow

```
1. Write the component module to the workspace, e.g. workspace/my-app.js
2. serve_local_data("my-app.js")           -> { url }   (CORS-served URL)
3. open_live_view(
     view_type="custom",
     title="My App",
     state={ "module_url": <url from step 2>, ...initial app state },
   )                                        -> view_id
```

`state.module_url` is REQUIRED for custom views — it tells the host which
component to load. Everything else in `state` is your component's own
initial state, delivered to `lv.onState`.

## Driving and observing it

- `live_view_update(view_id, patch)` — deep-merges `patch` into the state;
  the component's `onState` fires with the merged result.
- `live_view_call(view_id, action, args)` — invokes a `defineAction`
  handler; returns its result to you.
- `live_view_get_state(view_id)` — reads the current state, **including
  changes the user made** (whatever the component last `emitState`d).

## Complete example — a controllable scatter plot

`scatter-app.js` — renders points from state, click selects one (reported
back), and exposes a `highlight` action:

```js
export function setup(lv, root) {
  const W = 400, H = 300

  function render(state) {
    const points = state.points || []
    const selected = state.selected
    root.innerHTML = `
      <svg width="100%" height="100%" viewBox="0 0 ${W} ${H}"
           style="background:#111418">
        ${points.map((p, i) => `
          <circle cx="${p.x}" cy="${p.y}" r="${i === selected ? 9 : 5}"
            fill="${i === selected ? '#58a6ff' : '#8b949e'}"
            data-i="${i}" style="cursor:pointer" />
        `).join('')}
      </svg>`
    root.querySelectorAll('circle').forEach((el) => {
      el.addEventListener('click', () => {
        // user interaction → report new state outward
        lv.emitState({ ...lv.state, selected: Number(el.dataset.i) })
      })
    })
  }

  lv.onState(render)

  // agent can call: live_view_call(view_id, "highlight", {index: 2})
  lv.defineAction('highlight', ({ index }) => {
    lv.emitState({ ...lv.state, selected: index })
    return { selected: index }
  })
}
```

Open it:
```
serve_local_data("scatter-app.js")  ->  url
open_live_view("custom", "Scatter", {
  module_url: url,
  points: [{x:50,y:80}, {x:160,y:200}, {x:300,y:120}],
  selected: 0,
})
```

Then drive it — `live_view_update(view_id, {selected: 2})` moves the
selection; `live_view_get_state(view_id)` shows which point the user clicked.

## Tips

- Keep `onState` a pure render of the full state — idempotent, no surprises.
- Put everything the agent needs to control in `state`; expose discrete
  commands as actions.
- The component runs in a sandboxed iframe; it can fetch CORS-enabled URLs
  (use `serve_local_data` for workspace files) and import CDN libraries.
