/**
 * Multiple Sequence Alignment LiveView adapter.
 *
 * Wraps EBI Nightingale's <nightingale-msa> Web Component — modern, well
 * maintained, supports common protein color schemes (clustal, taylor,
 * hydro, zappo, etc.) and DNA alignments. Importing the module side-
 * effect-registers the custom element.
 *
 * LiveView "state":
 *   {
 *     "sequences": [{ "name": "seq1", "sequence": "ACGT-..." }, ...],
 *     "color_scheme": "clustal",   // optional: clustal | taylor | hydro |
 *                                  //   zappo | helix | strand | turn |
 *                                  //   buried | cinema | lesk | ...
 *     "tile_width":   20,          // px per column (default 20)
 *     "tile_height":  20,          // px per row    (default 20)
 *     "label_width":  80,          // px reserved for sequence names
 *     "display_start": 1,          // 1-based first visible column
 *     "display_end":   null,       // optional; defaults to alignment length
 *     "width":   null,             // optional; defaults to container width
 *     "height":  null              // optional; defaults to fit-all-rows
 *   }
 */
import 'https://esm.sh/@nightingale-elements/nightingale-msa@5.6.0'

export async function setup(lv, root) {
  root.style.width = '100%'
  root.style.height = '100%'
  root.style.overflow = 'auto'
  root.style.background = '#ffffff'
  root.style.color = '#222'

  let el = null
  let lastKey = null

  function applyState(state) {
    if (!state || !Array.isArray(state.sequences) || state.sequences.length === 0) {
      lv.fail('MSA: state must include a non-empty `sequences` array of '
        + '{name, sequence}.')
      return
    }
    // Validate equal lengths (alignment invariant).
    const seqLen = state.sequences[0].sequence.length
    const bad = state.sequences.find((s) => s.sequence.length !== seqLen)
    if (bad) {
      lv.fail(`MSA: sequences must be the same length; '${bad.name}' is `
        + `${bad.sequence.length} but expected ${seqLen}.`)
      return
    }

    const key = JSON.stringify([state.sequences, state.tile_width,
      state.tile_height, state.color_scheme, state.label_width,
      state.display_start, state.display_end, state.width, state.height])
    if (el && key === lastKey) return
    lastKey = key

    root.innerHTML = ''
    el = document.createElement('nightingale-msa')
    const w = state.width  || root.clientWidth  || 800
    const tileH = state.tile_height || 20
    const h = state.height || Math.min(800, Math.max(120, state.sequences.length * tileH + 32))
    el.setAttribute('width', String(w))
    el.setAttribute('height', String(h))
    el.setAttribute('tile-width',  String(state.tile_width  || 20))
    el.setAttribute('tile-height', String(tileH))
    el.setAttribute('label-width', String(state.label_width || 80))
    el.setAttribute('color-scheme', state.color_scheme || 'clustal')
    el.setAttribute('length', String(seqLen))
    el.setAttribute('display-start', String(state.display_start || 1))
    el.setAttribute('display-end',   String(state.display_end || seqLen))

    root.appendChild(el)
    // `data` is set as a JS property, not an HTML attribute.
    el.data = state.sequences.map((s) => ({
      name: s.name, sequence: s.sequence,
    }))
  }

  lv.onState((state, info) => {
    if (info && info.reason === 'emit') return
    try { applyState(state) }
    catch (e) { lv.fail('MSA: ' + ((e && e.message) || e)) }
  })

  // Nightingale renders SVG + canvas; the host's html2canvas catches the
  // SVG/DOM well enough. No custom provider needed.
}
