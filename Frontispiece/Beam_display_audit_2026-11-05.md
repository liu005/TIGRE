# Beam-display audit: 5 November 2026

All **15 Python GUI presets passed** the display audit in the `recon` Conda
environment. The original view counts and detector dimensions were retained,
including 5701 views for the GCT preset and 1253 for cargo DR.

## Scope

Each preset was loaded through the GUI, with beam controls exercised On and
Off. The first, middle and final views were checked for finite profile
coordinates, 20% opacity and envelopes ending 5% beyond the physical detector
along the beam axis. Off removed both profiles and aperture masking. Loading
another example reset the beam state. Only the diamond preset enables beams
by default; the other 14 start Off and use Toward origin when enabled.

The audit checked the diamond's segment-normal frames and the other presets'
rotating, fixed-detector, tilted, COR-offset, object-rotation and conveyor
configurations. The diamond included its hard aperture. Other presets were
enabled with their default contour and no hard aperture.

## Findings and corrections

1. A disabled aperture left a misleading Hard aperture entry in the legend.
   That artist is now created only when an aperture exists.
2. Beam envelopes extended beyond plot limits in every preset when enabled.
   Bounds now include full envelopes and physical detector pixels across the
   scan. Cached per-view lengths avoid repeating the detector-distance scan
   during playback. Scan-wide limits stay fixed while animating.

After these corrections, sampled views in every preset had no clipped
envelopes or empty aperture legend entries. Wider beams can make hardware
appear smaller because the complete envelope is included in the axes.

## Evidence and limits

- Machine-readable results and selected preview images:
  `.codex-results/beam_display_audit_20261104/`. The directory retains its
  original task-start date; the final rerun completed on 5 November.
- Original findings: `baseline_results.json`; corrected run: `results.json`.
- Focused beam, diamond and renderer checks: **30 passed**.

This is a software display audit. It does not establish measured source
profiles, hardware clearances, CT completeness or reconstruction quality.
Every view contributes to the plot bounds; visual/update checks sampled three
views per preset rather than manually inspecting every animation frame.
