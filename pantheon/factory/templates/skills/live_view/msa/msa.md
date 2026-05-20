---
id: msa_live_view
name: Multiple Sequence Alignment Viewer
description: |
  Open and drive a multiple-sequence-alignment view in the Pantheon
  sidebar. Built on EBI Nightingale's `<nightingale-msa>` Web Component —
  protein and nucleotide alignments with the standard colour schemes
  (clustal, taylor, hydro, zappo, ...) and configurable tile sizes.
tags: [msa, multiple-sequence-alignment, protein-alignment, dna-alignment, nightingale, live-view]
---

# Multiple Sequence Alignment Viewer

Show a pre-computed alignment (FASTA `.aln`, ClustalW, MUSCLE, MAFFT
output, etc.) as an interactive coloured matrix. Tip labels left,
residues right; pan + zoom horizontally for long alignments.

## When to use

- A protein or DNA alignment from MAFFT / MUSCLE / Clustal / MMseqs2 —
  inspect conservation visually.
- Compare a few orthologs at a key locus (Hb α across vertebrates,
  variants of a viral protein, ...).
- Side-by-side with phylogeny: `phylotree` for the tree, this for the
  alignment.

For *building* the alignment, run MAFFT / MUSCLE on a FASTA via shell.
For just one sequence's secondary-structure or features → use Nightingale's
other tracks (not wrapped here yet).

## Quick demo — built in

```
open_live_view(view_type="msa", title="Hb alpha across vertebrates")
```

No `state` → opens the bundled `demo.json` (a vertebrate haemoglobin
α-chain alignment, 5 species, ~140 columns, clustal coloring).

## The state — sequences + display options

```jsonc
{
  // REQUIRED — all sequences must be the same length (the alignment
  // invariant). Gaps are usually '-' or '.'.
  "sequences": [
    { "name": "Hb_alpha_human", "sequence": "VLSPADKTNVK--AAWGK..." },
    { "name": "Hb_alpha_mouse", "sequence": "VLSGEDKSNIK--AAWGK..." }
  ],

  // Optional display knobs (all have sensible defaults)
  "color_scheme":  "clustal",   // clustal | taylor | hydro | zappo |
                                //   helix | strand | turn | buried |
                                //   cinema | lesk | nucleotide
  "tile_width":    20,          // px per column
  "tile_height":   20,          // px per row
  "label_width":   80,          // px reserved for sequence names
  "display_start": 1,           // 1-based first visible column
  "display_end":   null,        // 1-based last visible column; defaults
                                //   to alignment length
  "width":         null,        // px; defaults to container width
  "height":        null         // px; defaults to fit-all-rows
}
```

## From data → state

### From a FASTA alignment file

```python
def read_fasta(path):
    seqs, name, buf = [], None, []
    with open(path) as fp:
        for line in fp:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append({"name": name, "sequence": "".join(buf)})
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
        if name is not None:
            seqs.append({"name": name, "sequence": "".join(buf)})
    return seqs

seqs = read_fasta("alignment.aln.fasta")
open_live_view(view_type="msa", title="My alignment",
               state={"sequences": seqs, "color_scheme": "clustal"})
```

### From an existing MAFFT/MUSCLE output

These tools write aligned FASTA — same as above. If output is in Clustal
or PHYLIP format, convert via `biopython`:

```python
from Bio import AlignIO
aln = AlignIO.read("alignment.aln", "clustal")
seqs = [{"name": rec.id, "sequence": str(rec.seq)} for rec in aln]
```

### Quick on-the-fly MAFFT

```python
import subprocess
subprocess.run(["mafft", "--auto", "input.fasta"],
               stdout=open("aligned.fasta", "w"), check=True)
# then read aligned.fasta as above
```

## Driving the view

Replace whole `sequences` with `live_view_set_state`:

```python
# zoom in on a 40-column window around the active site
live_view_set_state(view_id, {
    **current_state,
    "display_start": 58,
    "display_end":   97,
})
```

## Colour schemes — when to pick which

- **clustal** — classic alignment colors by physico-chemical category
  (default; good general-purpose for proteins)
- **taylor** — different palette, similar grouping
- **hydro** / **zappo** — by hydrophobicity (useful to spot membrane spans)
- **helix** / **strand** / **turn** — by Chou-Fasman secondary-structure
  propensity
- **nucleotide** — A/C/G/T colors for DNA/RNA alignments

## Verify it

`live_view_get_state` — `status: ready`, empty `diagnostics`. Common
failure: sequences of unequal length (the adapter calls `lv.fail` with the
offending name). `live_view_screenshot` uses html2canvas to capture the
SVG/canvas Nightingale draws.
