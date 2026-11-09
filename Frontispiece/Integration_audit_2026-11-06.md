# Operator and GUI split integration audit

This audit covers the current development worktree, not a released TIGRE
version. It focuses on the operator migration, the core/GUI boundary, manual
examples and integration tests. It is not a line-by-line proof of every
algorithm, a security review or validation on measured scanner data.

## Findings and changes

- The core Python export test was left behind after its implementation and
  complete test moved to tigre-geometry-design-tool. Its four tests failed with an undefined
  helper. The orphaned copy was removed; the GUI copy retains those tests.
- MATLAB's GUI exporter could not find `ctReconstructionTemplate.txt` because
  the template remained in TIGRE. It was moved next to the exporter in
  tigre-geometry-design-tool and the split manifest was updated. The GUI builder and the
  native exported CT scripts now pass their integration tests.
- The manual referred to an absent audit report, export script and figure.
  References were corrected; a manual export script and PDF formatting assets
  were added. The Python and MATLAB example blocks now have executable tests.
- Validation instructions included GUI tests no longer in TIGRE. The core
  commands now refer to retained tests, and a subprocess test guards recipe and
  exact geometry round trips against imports of Tk and tigre-geometry-design-tool.

## Evidence from this checkout

On 6 November 2026, using the `recon` environment, MATLAB R2020a, existing
local native binaries and an RTX 5070 Ti:

- The full core Python suite passed **615 tests**, with four pytest warnings
  and expected CUDA messages from error-path tests. This run included the new
  headless boundary test but preceded the new manual-example test; the latter
  also passed in a targeted run.
- The extracted GUI Python suite passed **210 tests**.
- MATLAB operator contract, all 26 operator/surface and cross-language cases,
  POCS, native surface projectors, surface FDK and geometry interchange passed.
  The flexible LSQR-TV cross-language relative difference was 0.000180466 in
  the isolated passing run, below the fixture's 0.002 threshold.
- MATLAB manual examples and seven GUI suites passed, including both native
  CT export smoke cases. A prior GUI run failed on the missing template before
  it was moved.
- An earlier MATLAB cross-language run failed an assertion in the flexible
  LSQR-TV case while the full Python suite was running concurrently. The
  assertion did not identify which of its checks failed. The isolated rerun
  and three additional isolated repeats passed; this does not establish the
  failure's cause or eliminate numerical variability. The test assertions now
  identify the failing check and numerical difference. Run GPU validation
  serially and investigate any recurrence.
- The Markdown manual generated a 16-page A4 PDF and an HTML edition using
  Pandoc and XeLaTeX. Relative links in the current manual and integration
  guides resolved in this workspace.

Commands and scope are in [Validation](Validation.md) and
[Projection operators](Projection_operator.md). The separate GUI checkout has
its own validation guide.

## Outstanding limitations

- The GUI repository is local and has no configured remote. Relative links to
  the sibling checkout will not resolve in a standalone published TIGRE
  checkout until a hosting location is chosen and links are updated.
- Both repositories ignore PDFs. Generated PDFs, including the two moved GUI
  PDFs, exist locally but will not be included by ordinary Git staging.
  Markdown sources and generation scripts are the reproducible sources.
- Existing changes across the large worktree remain uncommitted. `git diff
  --check` reports 266 whitespace findings in the tracked TIGRE diff; these
  were not normalized during the functional integration audit.
- Existing native binaries were used; no clean CUDA rebuild, multi-GPU or
  cross-platform validation was performed. Fixture results do not certify
  clinical use or quantitative accuracy on real acquisitions.
