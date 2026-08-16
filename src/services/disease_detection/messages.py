"""The Arabic advice sheet for the local model's fixed label space.

The local model answers from a closed set of 116 class names, so its advice is
looked up rather than generated: the text is reviewed once and served
identically every time. The remote providers have open label spaces and cannot
be served this way -- Gemini writes their message instead.

The sheet doubles as the class reference: every class the checkpoint can emit
has a key here, and the healthy ones map to null.
"""

import json
from pathlib import Path

_PATH = Path(__file__).parent / "data" / "disease_messages_ar.json"

# Loaded once at import. Healthy classes are present with a null value, which
# is the same answer an unknown key gets -- see message_for.
_MESSAGES: dict[str, str | None] = json.loads(_PATH.read_text(encoding="utf-8"))


def message_for(class_name: str) -> str | None:
    """The Arabic advice for a local-model class name, or None if there is none.

    None covers both a healthy class and a name absent from the sheet. Callers
    do not need to tell those apart: neither one has advice to give, and the
    startup check in YoloProvider is what surfaces an unexpected absence.
    """
    return _MESSAGES.get(class_name)


def known_classes() -> set[str]:
    """Every class name the sheet covers, healthy ones included."""
    return set(_MESSAGES)
