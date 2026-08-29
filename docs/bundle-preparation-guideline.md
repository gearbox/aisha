# Bundle preparation guideline

## 5. Workflow contract v2

Every `workflow:` map uses `contract_version: 2` and declares whether its graph is shaped for
`media: image` or `media: video`. The map binds Apex's canonical request parameters to the actual
ComfyUI input names; the left-hand side is always canonical and the right-hand side is the API
input key.

```yaml
workflow:
  contract_version: 2
  media: video
  nodes:
    latent:
      id: "9"
      class: EmptyHunyuanLatentVideo
      inputs: {width: width, height: height, length: length}
    positive_prompt:
      id: "3"
      class: CLIPTextEncode
      inputs: {text: text}
    sampler:
      id: "2"
      class: KSampler
      inputs: {seed: seed, steps: steps}
  media_inputs:
    - id: "7"
      class: LoadImage
      input: image
      kind: image
      slot: first_frame
      target_role: latent
      target_input: image
```

`media_inputs` identifies the loader node, the loader input that receives the uploaded filename,
and the role input linked from its output. `input` defaults to `image` for the image loaders, and
`target_role` defaults to `positive_prompt`; every other field is explicit. Slots describe request
capability rather than graph position:

| Slot | Required `kind` | Capability |
| --- | --- | --- |
| `reference` | `image` | Image reference / edit conditioning; multiple references are allowed. |
| `first_frame` | `image` | Image-to-video conditioning. |
| `last_frame` | `image` | First-and-last-frame video conditioning; requires an image `first_frame`. |
| `source` | `video` | Video-to-video source media. |

One `media_inputs` entry represents one uploaded asset and one loader node. A loader may feed
additional graph consumers; those consumers receive that same uploaded asset and must not be
duplicated as additional `media_inputs` entries. Supported loaders have fixed semantics: `LoadImage`
and `LoadImageMask` accept an image at `input: image`; `LoadVideo` accepts a video at `input: file`;
and `VHS_LoadVideo` accepts a video at `input: video`. The declared `kind`, loader input, and target
edge must match those semantics. In particular, `LoadImage` target edges use output slot `0` (image),
never slot `1` (mask).

`kind` describes the uploaded asset accepted by Apex, not necessarily the ComfyUI tensor emitted by
the loader. For example, `LoadImageMask` still has `kind: image` because Apex uploads an image file
to its `image` input, even though that loader's contract edge is its mask output at slot `0`.

Use `length`, `fps`, and `format` only for `media: video`. `model_sampling.shift` is a separate
role from `sampler`; do not place it on `sampler`. A media target cannot reuse an API input already
mapped as a scalar role parameter.

Every model group with a known loader must be bound through a corresponding `workflow.model_inputs`
entry. Generate or refresh this block without a ComfyUI installation or downloaded weights with
`acs workflow map --api workflow.api.json --bundle bundle.yaml`; add `--write` only after reviewing
the rendered output. The validator reports these model-binding problems:

- `workflow.model.loader_unmapped`: a writable recognized loader in the API graph has no binding.
- `workflow.model.loader_partially_mapped`: only some same-class loaders are bound; confirm that
  the remaining loaders are deliberately fixed.
- `workflow.model.group_missing`: a declared binding names a model type not present in `models:`.
- `workflow.model.binding_input_mismatch`: a `model_inputs` entry's `input` does not match the
  writable input of the loader class actually at that node id (e.g. `input: weight_dtype` on a
  `UNETLoader`, whose writable input is `unet_name`). The binding does nothing; Apex never touches
  the field it names.
- `workflow.model.binding_type_mismatch`: a `model_inputs` entry's `model_type` does not match the
  model type of the loader class actually at that node id (e.g. `model_type: vae` on a
  `UNETLoader`, whose model type is `diffusion_models`). Apex resolves the wrong `models:` group for
  this node.

Before publishing, run `acs bundle validate <name>` and deliberately verify these failures offline:

1. Add `latent.inputs.length` to an image workflow; expect
   `workflow.media.parameter_cross_media`.
2. Replace a media target link in `workflow.api.json` with a scalar; expect
   `workflow.media.target_not_linked`.
3. Change `contract_version` to `1`; expect `workflow.contract.version_unsupported`.
