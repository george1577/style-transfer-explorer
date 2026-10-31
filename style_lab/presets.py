"""Style presets: a named target style with its instructions, default input, example pairs and checklist.

Example pairs come from a source function, so each preset can get them from wherever fits:
social media uses notebook 1's four pairs; Professional email draws them, in fixed order, from the
dataset's train split (so test pairs never reach a prompt).
"""
from collections.abc import Callable
from dataclasses import dataclass

from style_lab.checklist import Check, email_checklist, social_media_checklist
from style_lab.dataset import train_test_split
from style_lab.prompts import ExamplePair


@dataclass(frozen=True)
class StylePreset:
    name: str
    instructions: str
    default_input: str
    example_pairs: Callable[[], list[ExamplePair]]
    checklist: Callable[[str, str], list[Check]]


# ---- Professional email (notebook 0, dataset system prompt) ---------------------------

EMAIL_INSTRUCTIONS = "You are a helpful assistant converting notes to professional emails."

# Notebook 0's meeting-minutes notes, with the source's leading newline and stray space removed.
EMAIL_DEFAULT_INPUT = """urgent: need meeting minutes by {deadline}
for stakeholder presentation
details:
- This is needed for our upcoming presi to stakeholders.
- will need ur help asap
- send 2 me when rdy"""


def email_train_pairs() -> list[ExamplePair]:
    return list(train_test_split().train)


# ---- Social media (notebook 1) --------------------------------------------------------

SOCIAL_MEDIA_INSTRUCTIONS = "Convert the following text into social media styled text:"

SOCIAL_MEDIA_PAIRS = [
    ExamplePair("The movie was fantastic!",
                "OMG the camera work was everything! The lighting was so mood, I can't even 😍"),
    ExamplePair("I enjoyed the concert last night.",
                "Last night's concert was FIRE 🔥🔥 Totally vibing to those beats! #blessed #musicislife"),
    ExamplePair("The new restaurant has great food.",
                "This new spot is GOALS 😋 The food? I'm obsessed! Straight up food porn for your feed #foodie #noms"),
    ExamplePair("The hiking trail was beautiful and peaceful.",
                "Just crushed this epic trail! 🏞️ Views for dayssss! Nature therapy is SO underrated #adventure #nofilter"),
]


def social_media_pairs() -> list[ExamplePair]:
    return list(SOCIAL_MEDIA_PAIRS)


PRESETS: dict[str, StylePreset] = {
    p.name: p
    for p in [
        StylePreset("Professional email", EMAIL_INSTRUCTIONS, EMAIL_DEFAULT_INPUT,
                    example_pairs=email_train_pairs, checklist=email_checklist),
        StylePreset("Social media", SOCIAL_MEDIA_INSTRUCTIONS, "The sunset was beautiful this evening.",
                    example_pairs=social_media_pairs, checklist=social_media_checklist),
    ]
}
