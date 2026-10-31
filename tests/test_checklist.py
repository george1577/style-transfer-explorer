"""Seam B: the style checklist, as pure functions on (input, output)."""
from style_lab.checklist import email_checklist, social_media_checklist

NOTES = """urgent: need meeting minutes by {deadline}
for stakeholder presentation
details:
- This is needed for our upcoming presi to stakeholders.
- will need ur help asap
- send 2 me when rdy"""

PERFECT_EMAIL = """Subject: Meeting minutes needed by {deadline}

Body: Hello {name},

I need the meeting minutes by {deadline} for the stakeholder presentation.
- This is needed for our upcoming presi to stakeholders.
- Will need ur help ASAP.
- Send 2 me when rdy!

Kind regards,
{name}"""


def results(checks):
    return {c.name: c.passed for c in checks}


def check(checks, name):
    return next(c for c in checks if c.name == name)


def test_perfect_email_passes_every_check():
    checks = email_checklist(NOTES, PERFECT_EMAIL)
    assert results(checks) == {
        "Has Subject: line": True,
        "Has Body: section": True,
        "Greeting": True,
        "Sign-off with {name}": True,
        "Content kept": True,
    }
    assert all(c.evidence for c in checks)


def test_missing_subject_fails_only_subject_check():
    email = PERFECT_EMAIL.replace("Subject: Meeting minutes needed by {deadline}\n\n", "")
    r = results(email_checklist(NOTES, email))
    assert r["Has Subject: line"] is False
    assert r["Has Body: section"] and r["Greeting"] and r["Content kept"]


def test_missing_body_label_fails_body_check():
    email = PERFECT_EMAIL.replace("Body: Hello", "Hello")
    checks = email_checklist(NOTES, email)
    assert results(checks)["Has Body: section"] is False
    assert results(checks)["Greeting"] is True


def test_missing_greeting_fails_greeting_check():
    email = PERFECT_EMAIL.replace("Body: Hello {name},\n\n", "Body: ")
    assert results(email_checklist(NOTES, email))["Greeting"] is False


def test_greeting_evidence_quotes_line_and_reports_name_placeholder():
    with_name = check(email_checklist(NOTES, PERFECT_EMAIL), "Greeting")
    assert "Hello {name}," in with_name.evidence
    assert "uses {name}" in with_name.evidence

    team = check(email_checklist(NOTES, PERFECT_EMAIL.replace("Hello {name},", "Dear Finance team,")), "Greeting")
    assert team.passed
    assert "Dear Finance team," in team.evidence
    assert "does not use {name}" in team.evidence


def test_missing_sign_off_fails_sign_off_check():
    email = PERFECT_EMAIL.replace("\n\nKind regards,\n{name}", "")
    assert results(email_checklist(NOTES, email))["Sign-off with {name}"] is False


def test_sign_off_without_name_placeholder_fails():
    email = PERFECT_EMAIL.replace("Kind regards,\n{name}", "Best regards,\n[Your Name]")
    assert results(email_checklist(NOTES, email))["Sign-off with {name}"] is False


def test_dropped_bullet_fails_content_kept_and_names_it():
    email = PERFECT_EMAIL.replace("- Will need ur help ASAP.\n", "")
    kept = check(email_checklist(NOTES, email), "Content kept")
    assert kept.passed is False
    assert "will need ur help asap" in kept.evidence


def test_bullets_match_despite_case_punctuation_and_whitespace():
    email = PERFECT_EMAIL.replace("- Send 2 me when rdy!", "send   2 ME, when rdy.")
    assert results(email_checklist(NOTES, email))["Content kept"] is True


def test_notes_without_bullets_treat_each_line_as_a_point():
    notes = "feedback 4 mobile app\nReviewed current progress"
    email = PERFECT_EMAIL + "\nReviewed current progress"
    kept = check(email_checklist(notes, email), "Content kept")
    assert kept.passed is False
    assert "feedback 4 mobile app" in kept.evidence

    kept_all = check(email_checklist(notes, email + "\nfeedback 4 mobile app"), "Content kept")
    assert kept_all.passed is True


def test_star_bullet_and_numbered_points_are_recognised():
    notes = "header line\n* first point\n• second point\n1. third point"
    email = "first point. second point. third point."
    assert results(email_checklist(notes, email))["Content kept"] is True


def test_social_media_counts_emoji_and_hashtags():
    post = "Last night's concert was FIRE 🔥🔥 Totally vibing! 🏞️ #blessed #musicislife"
    checks = social_media_checklist("I enjoyed the concert last night.", post)
    emoji = check(checks, "Emoji")
    hashtags = check(checks, "Hashtags")
    assert emoji.passed and emoji.evidence.startswith("3 emoji")
    assert hashtags.passed and hashtags.evidence.startswith("2 hashtags")
    assert "#blessed" in hashtags.evidence


def test_plain_text_fails_social_media_checks():
    checks = social_media_checklist("x", "The concert was good. Item #3 was best.")
    assert results(checks) == {"Emoji": False, "Hashtags": False}
    assert check(checks, "Emoji").evidence.startswith("0 emoji")
