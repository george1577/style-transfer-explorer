# Style Transfer Explorer

A learning tool for seeing how a generative model can be made to rewrite text or images in a target style, and what each technique costs.

## Language

### Core ideas

**Style transfer**:
Rewriting an input so that its content is kept and its style matches a target.
_Avoid_: Restyling, conversion

**Content**:
What must be kept from the input: the facts in a text, or the composition (layout and depth) of an image.
_Avoid_: Substance, meaning

**Style**:
Everything a style transfer is allowed to change: tone, format, wording, or surface look.
_Avoid_: Voice (only one aspect of style), tone

**Style checklist**:
A set of rule-based checks on an output that report which style features it shows and whether the content was kept.
_Avoid_: Score, eval, grade

**Test pair**:
An example pair held out from fine-tuning and few-shot prompts, used only to compare techniques fairly.
_Avoid_: Validation sample, eval example

### Text techniques

**Zero-shot**:
Getting a target style by describing it in instructions alone, with no examples.
_Avoid_: Plain prompting

**Few-shot**:
Getting a target style by putting input→output example pairs in the prompt before the new input.
_Avoid_: In-context learning (the broader idea), example prompting

**Fine-tuning**:
Getting a target style by training a model's weights on many input→output examples, so that at use time the prompt needs neither long instructions nor examples.
_Avoid_: Training (too broad), customization

**Base model**:
The pretrained model before any fine-tuning.
_Avoid_: Vanilla model, generic model

**Adapter**:
The small set of trained weights that, added to the base model, produces the fine-tuned model.
_Avoid_: LoRA (the method, not the artifact), checkpoint

**Example pair**:
One input together with its correct output in the target style.
_Avoid_: Shot, sample, demonstration

**Style preset**:
A named target style with its instructions and example pairs, e.g. Professional email or Social media.
_Avoid_: Task, mode

### Image technique

**Reference image**:
The input image whose composition the generated image must keep.
_Avoid_: Source image, init image, content image

**Depth map**:
A grayscale image estimating how far each pixel of the reference image is from the camera; it carries the composition into generation.
_Avoid_: Control image (the pipeline's name for it), mask

**Depth conditioning**:
Steering image generation with a depth map so the output keeps the reference image's composition.
_Avoid_: ControlNet (the model that implements it)

**Strength**:
How far generation is allowed to move away from the reference image's pixels, from 0 (unchanged) to 1 (ignored).
_Avoid_: Denoising strength, noise level

**Guidance**:
How closely generation follows the text prompt rather than drifting freely.
_Avoid_: CFG, prompt weight
