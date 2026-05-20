/**
 * RDKit-JS LiveView adapter — 2D small-molecule depiction.
 *
 * Complements the molstar 3D viewer with the canonical 2D drawing of
 * SMILES / MOL block input. Built from RDKit's C++ via WebAssembly —
 * the first call to `initRDKitModule` downloads ~3 MB of WASM, then
 * subsequent renders are instant.
 *
 * LiveView "state":
 *   {
 *     "molecules": [
 *       { "smiles":   "CCO",                             "name": "Ethanol" },
 *       { "smiles":   "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    "name": "Caffeine" },
 *       { "molblock": "<MOL file string>",               "name": "From MOL" }
 *     ],
 *     "draw_options": {                                  // optional
 *       "width": 320, "height": 220,
 *       "addAtomIndices": false,
 *       "highlightAtoms": [],         // 0-based indices to highlight
 *       "highlightBonds": []
 *     }
 *   }
 *
 * Renders a grid of SVG depictions, one per molecule, with the name
 * underneath. Invalid SMILES are shown as red error cards in-line.
 */
import initRDKitModule from 'https://esm.sh/@rdkit/rdkit@2025.3.4-1.0.0'

export async function setup(lv, root) {
  root.style.width = '100%'
  root.style.height = '100%'
  root.style.overflow = 'auto'
  root.style.background = '#ffffff'
  root.style.color = '#222'
  root.style.padding = '12px'
  root.style.fontFamily = 'system-ui, -apple-system, sans-serif'

  root.innerHTML = '<div style="padding:24px;color:#666;font-size:13px">'
    + 'Loading RDKit (WASM, ~3 MB)…</div>'

  let RDKit
  try {
    RDKit = await initRDKitModule()
  } catch (e) {
    lv.fail('RDKit: failed to load WASM — ' + ((e && e.message) || e))
    return
  }

  let lastKey = null

  function applyState(state) {
    if (!state || !Array.isArray(state.molecules) || state.molecules.length === 0) {
      lv.fail('RDKit: state must include a non-empty `molecules` array '
        + 'of {smiles, name?} or {molblock, name?}.')
      return
    }
    const key = JSON.stringify([state.molecules, state.draw_options])
    if (key === lastKey) return
    lastKey = key

    const opts = state.draw_options || {}
    const W = opts.width  || 320
    const H = opts.height || 220

    // RDKit's drawing options JSON — used by get_svg_with_highlights.
    const drawDetails = JSON.stringify({
      width: W, height: H,
      addAtomIndices: !!opts.addAtomIndices,
      atoms: opts.highlightAtoms || [],
      bonds: opts.highlightBonds || [],
    })

    root.innerHTML = ''
    const grid = document.createElement('div')
    grid.style.display = 'grid'
    grid.style.gridTemplateColumns = `repeat(auto-fit, minmax(${W + 8}px, 1fr))`
    grid.style.gap = '16px'

    for (const m of state.molecules) {
      const cell = document.createElement('div')
      cell.style.display = 'flex'
      cell.style.flexDirection = 'column'
      cell.style.alignItems = 'center'
      cell.style.gap = '6px'

      let html = ''
      let mol = null
      try {
        mol = m.molblock
          ? RDKit.get_mol(m.molblock)
          : RDKit.get_mol(String(m.smiles || ''))
        if (mol && mol.is_valid()) {
          // get_svg_with_highlights handles size + highlights uniformly.
          html = mol.get_svg_with_highlights(drawDetails)
        } else {
          html = '<div style="color:#f85149;font-size:12px;padding:8px">'
            + 'invalid input: ' + (m.smiles || '(molblock)') + '</div>'
        }
      } catch (e) {
        html = '<div style="color:#f85149;font-size:12px;padding:8px">'
          + 'RDKit error: ' + ((e && e.message) || e) + '</div>'
      } finally {
        // WASM-allocated objects must be deleted explicitly to free memory.
        if (mol) { try { mol.delete() } catch (_) {} }
      }

      const svgBox = document.createElement('div')
      svgBox.innerHTML = html
      cell.appendChild(svgBox)

      const label = document.createElement('div')
      label.textContent = m.name || m.smiles || ''
      label.style.fontSize = '12px'
      label.style.color = '#444'
      label.style.maxWidth = W + 'px'
      label.style.textAlign = 'center'
      label.style.overflow = 'hidden'
      label.style.textOverflow = 'ellipsis'
      label.style.whiteSpace = 'nowrap'
      cell.appendChild(label)

      grid.appendChild(cell)
    }
    root.appendChild(grid)
  }

  lv.onState((state, info) => {
    if (info && info.reason === 'emit') return
    try { applyState(state) }
    catch (e) { lv.fail('RDKit: ' + ((e && e.message) || e)) }
  })

  // Pure SVG output — html2canvas captures it cleanly.
}
