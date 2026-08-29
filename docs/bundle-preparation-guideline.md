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
and the role input linked from its output. `input` defaults to `image`, and `target_role` defaults
to `positive_prompt`; every other field is explicit. Slots describe request capability rather than
graph position:

| Slot | Capability |
| --- | --- |
| `reference` | Image reference / edit conditioning; multiple references are allowed. |
| `first_frame` | Image-to-video conditioning. |
| `last_frame` | First-and-last-frame video conditioning; requires `first_frame`. |
| `source` | Video-to-video source media. |

Use `length`, `fps`, and `format` only for `media: video`. `model_sampling.shift` is a separate
role from `sampler`; do not place it on `sampler`. A media target cannot reuse an API input already
mapped as a scalar role parameter.

Before publishing, run `acs bundle validate <name>` and deliberately verify these failures offline:

1. Add `latent.inputs.length` to an image workflow; expect
   `workflow.media.parameter_cross_media`.
2. Replace a media target link in `workflow.api.json` with a scalar; expect
   `workflow.media.target_not_linked`.
3. Change `contract_version` to `1`; expect `workflow.contract.version_unsupported`.
