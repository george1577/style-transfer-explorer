from style_lab.presets import PRESETS


def test_email_preset_defaults_to_notebook_meeting_minutes_notes():
    email = PRESETS["Professional email"]
    assert email.instructions == "You are a helpful assistant converting notes to professional emails."
    assert email.default_input.startswith("urgent: need meeting minutes by {deadline}")
    assert email.default_input.endswith("- send 2 me when rdy")


def test_social_media_preset_offers_notebook_example_pairs_in_order():
    pairs = PRESETS["Social media"].example_pairs()
    assert [p.input for p in pairs] == [
        "The movie was fantastic!",
        "I enjoyed the concert last night.",
        "The new restaurant has great food.",
        "The hiking trail was beautiful and peaceful.",
    ]
    assert pairs[1].output == "Last night's concert was FIRE 🔥🔥 Totally vibing to those beats! #blessed #musicislife"


def test_email_preset_offers_train_pairs_in_fixed_order_and_never_test_pairs():
    from style_lab.dataset import train_test_split

    split = train_test_split()
    pairs = PRESETS["Professional email"].example_pairs()

    assert pairs == list(split.train)
    assert pairs[0].input.startswith("feedback 4 mobile app\n")
    assert not set(pairs) & set(split.test)
