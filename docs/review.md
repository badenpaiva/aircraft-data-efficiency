# Generated-image review and bounding boxes

## Current policy

The automatic SUTD pipeline uses only candidates whose technical screening outcome is `pass`. It proposes non-overlapping placements on real subset images and labels each pasted patch by its core: the part that the GAN's training crops filled with the annotated box (see [HQGAN design](hqgan.md#crops-placement-and-labels)). With the current crop margin the core is the central 80% of the patch; the completed first run had no margin, so its label was the whole patch, `[0, 0, 1, 1]`. Original image labels are retained. No human approvals are created, and final composites are not semantically reviewed. These are approximate training labels.

`pass` checks image integrity, dimensions, contrast, detail and duplication. It does not certify that the defect is present, the class is correct or the box is tight. `needs_review` candidates are excluded by the current whole-patch entry point. `reject` candidates cannot be approved by the reviewed workflow.

## Refining review and boxes

The supported review workflow changes candidate selection and annotations while retaining the SUTD preparation, subset-bound GANs, blending, detector training and evaluation pipeline. Review a candidate's defect/class and tight box, inspect placement, then inspect the blended image and all labels. CVAT exchanges below provide those decisions without typing normalized coordinates.

Use the source study's review queues, not the current derived whole-patch manifests. The `whole_patch_study.py` entry point always creates crop-rule labels even if a queue has human boxes; reviewed composites are consumed by `detection_study.py run`. Generated patches are stored with their long side horizontal. Draw the box on the patch as shown; a placement on a tall real box turns the patch and its box a quarter turn automatically. Do not modify inputs of completed detector runs. Existing source candidate queues can be reviewed without retraining their GANs if their provenance remains valid.

After candidate review for a subset:

```powershell
.\.venv\Scripts\python.exe scripts/detection_study.py propose-placements --study results/studies/sutd_units_v2 --subset seed0_f02500
```

After placement review:

```powershell
.\.venv\Scripts\python.exe scripts/detection_study.py blend --study results/studies/sutd_units_v2 --subset seed0_f02500
```

After final composite review:

```powershell
.\.venv\Scripts\python.exe scripts/detection_study.py run --study results/studies/sutd_units_v2 --arms hqgan_0p5 hqgan_1p0 hqgan_2p0
```

Repeat review for each intended subset. The run command checks all planned seeds/fractions; subsets without sufficient reviewed composites remain blocked. Placements currently number exactly the largest ratio's quota, so rejected placements/composites can leave that ratio short. There is no automatic replacement round. Do not assume whole-patch and reviewed results are interchangeable.

Explicit `screen --stage candidates` or `screen --stage composites` resets that stage's human decisions. Placement/blend commands refuse to overwrite existing outputs. Review changes requiring regeneration need new output provenance rather than edits to frozen inputs.

CVAT integration is file-based: export locally, upload into CVAT, export annotations and import locally. No server, upload or credentials are managed by these commands. Use CVAT for images 1.1, which preserves review tags and attributes; YOLO/COCO exports do not preserve this exchange metadata.

## 1. Export a candidate review batch

First run the `gan` command from the [pipeline guide](pipeline.md). New candidate queues are screened
automatically. Then, from the repository root in PowerShell:

```powershell
$studyPath = 'results/studies/sutd_units_v2'
$subsetKey = 'seed0_f10000'
$packagePath = "$studyPath/cvat/$subsetKey/candidates_batch01"

.\.venv\Scripts\python.exe scripts/detection_study.py cvat-export --study $studyPath --subset $subsetKey --stage candidates --output $packagePath --limit 100
```

Use your actual study path and subset key. The output directory must be new.
`--limit` is optional; it selects up to that many undecided items. Already reviewed
items and technical rejections are excluded. A flagged item remains eligible for
human inspection, but approving it requires an explanation.

The package contains:

| File | Use |
|---|---|
| `images.zip` | Image data to upload when creating the CVAT task |
| `labels.json` | CVAT task label/attribute definitions |
| `annotations.xml` | Initial per-image review tags and reference boxes |
| `exchange.json` | Local mapping, dimensions, queue identity and content hashes; retain unchanged |

No candidate box is guessed or supplied initially. Images retain their original
sizes, with unique exchange filenames mapped back to pipeline IDs.

## 2. Create and annotate the task in CVAT

1. Create a new **image task**. In its label editor's **Raw** view, paste the
   contents of `labels.json` and apply them. Use the supplied schema unchanged.
2. Upload `images.zip` as the task's data. Include every image; do not subsample,
   rename or resize them. Use high image quality for examining subtle defects.
3. After task creation, import/upload `annotations.xml` using **CVAT for images
   1.1**. This initializes the `review` tag on each image.
4. Open a job. On each frame, inspect its existing `review` tag in the objects
   panel. The `context` field identifies the expected defect class. The
   `screening` field contains the technical outcome and flags.

CVAT's [task creation guide](https://docs.cvat.ai/docs/workspace/tasks-page/)
describes its label constructor, Raw JSON editor and image upload. Menu wording
may differ by CVAT version. The task must retain the supplied tag attributes.

For a **candidate you accept**:

- Draw exactly one axis-aligned rectangle around the visible defect, using the
  expected class label (`missing-rivet`, `rust` or `scratch`). Zoom in as needed.
- On the existing `review` image tag, check `defect_present` and `class_correct`.
- Set `decision` to `approve`.
- If screening flagged the image, enter a specific justification in `note`.

For an unsuitable image, set `decision=reject`; a box and confirmation checkboxes
are not required. Use `note` to explain the rejection. Leave uncertain/uninspected
images at `pending`. Do not add another review tag or change `exchange_id`,
`context`, or `screening`. Marking the CVAT job complete does not itself approve
images: the importer reads each image's explicit decision.

The semantic criteria are recognizable defects, correct classes and plausible imagery. Reject blank/noisy patches,
wrong-class defects, implausible imagery or patches that cannot be adequately
represented by one defect box. A drawn box alone is never interpreted as approval.

## 3. Bring the decisions back

Save your work, then export the **whole task's annotations** as **CVAT for images
1.1**. You can turn off image export. Keep all task frames, including rejected and
pending ones. The local importer accepts the downloaded ZIP containing
`annotations.xml`, or the extracted XML itself.

```powershell
$annotationFile = 'D:/Downloads/cvat_candidates_batch01.zip'

# Validate the entire batch without writing any decisions.
.\.venv\Scripts\python.exe scripts/detection_study.py cvat-import --study $studyPath --subset $subsetKey --stage candidates --package $packagePath --annotations $annotationFile --reviewer 'Your name' --dry-run

# Apply that same batch atomically after validation succeeds.
.\.venv\Scripts\python.exe scripts/detection_study.py cvat-import --study $studyPath --subset $subsetKey --stage candidates --package $packagePath --annotations $annotationFile --reviewer 'Your name'
```

Use the actual downloaded file path. `--reviewer` names the person who made the
batch's decisions. Use separate batches for separate reviewers. The import prints
approved/rejected/pending counts, converts candidate boxes automatically, and
writes decisions into the existing queue used by the pipeline. It keeps a
`*.before_cvat_*.json` backup and records the annotation-file hash in `cvat_imports`.

All checks pass before any decision is committed. A wrong class, missing image,
stale queue, changed image/checkpoint, invalid box or incomplete confirmation
blocks the whole batch. Fix the CVAT annotations and export again; do not alter
hashes or remove validation checks. A `needs_review` approval requires a nonempty
note. Technical rejects cannot be bypassed through CVAT.

A package is tied to the queue at export time. Once an import changes that queue,
the same package cannot be imported again. To continue, export a **new package**
(e.g. `candidates_batch02`) and create a new CVAT task for the remaining undecided
items. Pending images remain undecided and can be included in that next batch.
Do not run local review or re-screening commands while a CVAT batch is in progress;
they invalidate its snapshot. Work on one batch per stage/subset at a time.

## 4. Review placements in CVAT

After sufficient candidate approvals, run `propose-placements` as usual, then:

```powershell
$placementPackage = "$studyPath/cvat/$subsetKey/placements_batch01"
.\.venv\Scripts\python.exe scripts/detection_study.py cvat-export --study $studyPath --subset $subsetKey --stage placements --output $placementPackage --limit 100
```

Create another CVAT task with this package's labels, images and initial annotations.
Each background has a supplied rectangle labelled `placement`. This marks the
proposed insertion region. The tag's `context` field identifies the generated
candidate and its filename in `candidate_references.zip`; extract that separate
archive locally and inspect the corresponding patch alongside the background.
Do not upload the reference archive as extra frames in the review task.

Approve only suitable aircraft-skin locations at plausible scale/orientation.
Set `placement_valid=true` and `decision=approve`, or reject the placement.
**Do not move or resize this reference rectangle.** This integration reviews the
existing proposal; geometry changes are rejected on import. Reject unsuitable
proposals and regenerate through the documented workflow instead.

Export the entire CVAT task and use `cvat-import` with `--stage placements` and
`--package $placementPackage`. Use `--dry-run` first, as for candidate imports.
Then run `blend` to produce and automatically screen the composites.

## 5. Review final composites in CVAT

```powershell
$compositePackage = "$studyPath/cvat/$subsetKey/composites_batch01"
.\.venv\Scripts\python.exe scripts/detection_study.py cvat-export --study $studyPath --subset $subsetKey --stage composites --output $compositePackage --limit 100
```

Create a third task from that package. Existing boxes show **all** target-class
defects, including the inserted one. Inspect the actual blended image for a
recognizable defect, plausible appearance and complete/correct labels.

Set `realistic=true`, `labels_complete=true`, and `decision=approve` only when all
criteria hold. Explain flagged approvals in `note`. Otherwise reject or leave
pending. Do not add, remove, relabel or move boxes: this stage approves the frozen
composite and labels. Incorrect geometry needs regeneration/re-screening, not a
silent change to the experiment's training targets. Import with
`--stage composites --package $compositePackage`.

Detector training now consumes the approved composites through the existing
study runner. Nothing changes about subset isolation, synthetic ratios, review
requirements, or reusing the approved pool across supported YOLO models.

## Validation and limits

The integration has local round-trip tests for all three stages, coordinate
normalization, explicit decisions, ZIP/XML import, stale data, changed geometry,
missing/duplicate images and atomic failure. The format and label workflow were
checked against the official CVAT documentation. It has **not yet been exercised
against your CVAT server**. Start with a small batch and validate the exported
annotations with `--dry-run` before committing decisions.

No CVAT server is installed or configured by these commands. No API credentials
are stored, and no files are uploaded automatically. The original numeric `review`
CLI remains available, but must not be mixed with an outstanding CVAT exchange.
