"""CPU unit tests for the reasoning-effort / thinking probe (issue #97).

Faithful port of upstream's ``tests/tokenizer/test_effort.py`` to the in-process
seam: the probe is driven by a ``render(kwargs, tools) -> str`` callback and the
quantizer by the static ``EFFORT_SCALE`` table. Pure text -- no AutoTokenizer,
no Jinja, no torch -- so these run in the lean CPU venv and the dual-venv
"torch must not import" contract is unaffected. A real-tokenizer probe
round-trip is exercised on the XPU live suite (``test_serve_live_engine_xpu``).
"""
from __future__ import annotations

from freetoken.tokenizer.effort import (
    EFFORT_SCALE,
    EffortProfile,
    KNOWN_REASONING_EFFORTS,
    OPENAI_EFFORT_TRIPLE,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)


# --- Representative profiles (mirrors upstream test fixtures) ---------------
QWEN38 = EffortProfile(supported=frozenset({"xhigh", "medium", "low"}), default="xhigh", consumes_effort=True)
DSV4_OFFICIAL = EffortProfile(supported=frozenset({"low", "high", "max"}), default="low", consumes_effort=True)
IGNORES = EffortProfile(supported=frozenset(KNOWN_REASONING_EFFORTS), default=None, consumes_effort=False)


# --------------------------------------------------------------------------- #
# quantize_effort
# --------------------------------------------------------------------------- #
def test_in_vocabulary_values_pass_through():
    for name in ("xhigh", "medium", "low"):
        assert quantize_effort(name, QWEN38) == name
    for name in ("low", "high", "max"):
        assert quantize_effort(name, DSV4_OFFICIAL) == name


def test_deepseek_dialect_high_lands_on_qwen_xhigh():
    # high (0.9) is nearer xhigh (0.99) than medium (0.7)
    assert quantize_effort("high", QWEN38) == "xhigh"


def test_openai_dialect_medium_drops_to_the_dsv4_default():
    # medium (0.7) is 0.2 from the nearest gear -- beyond the quantize
    # threshold, so nothing is sent and the encoder default (low) applies;
    # anything else would silently escalate OpenAI-default traffic to the
    # encoder's absolute-maximum "high" prompt. Matches vLLM's DSV4 mapping.
    assert quantize_effort("medium", DSV4_OFFICIAL) is None


def test_xhigh_lands_on_dsv4_high_never_max():
    # "max" is an extreme opt-in gear, reachable only by its own name
    # (vLLM's DSV4 rule) -- rounding must not enter it.
    assert quantize_effort("xhigh", DSV4_OFFICIAL) == "high"


def test_max_lands_on_qwen_xhigh():
    # "max" is a removable default gear (xhigh==max on the scale): it rounds to
    # the nearest non-"max" gear, never to itself.
    assert quantize_effort("max", QWEN38) == "xhigh"


def test_far_values_drop_instead_of_rounding():
    profile = EffortProfile(supported=frozenset({"low", "xhigh"}), default=None, consumes_effort=True)
    assert quantize_effort("minimal", profile) == "low"  # 0.1 away
    assert quantize_effort("medium", profile) is None  # 0.29 to the nearest gear


def test_unknown_and_non_string_values_drop_to_template_default():
    assert quantize_effort("banana", QWEN38) is None
    assert quantize_effort(3, QWEN38) is None
    assert quantize_effort(None, QWEN38) is None


def test_effort_ignoring_template_sends_nothing():
    assert quantize_effort("high", IGNORES) is None
    assert quantize_effort("xhigh", IGNORES) is None


# --------------------------------------------------------------------------- #
# probe_effort_profile: fake render callables standing in for real templates
# --------------------------------------------------------------------------- #
def _qwen38_render(kwargs, tools):
    # Validates unconditionally (enable_thinking undefined counts as on) and
    # renders a distinct effort preamble per gear, default xhigh.
    effort = kwargs.get("reasoning_effort", "xhigh")
    if effort not in ("xhigh", "medium", "low"):
        raise ValueError(f"Unexpected reasoning effort {effort}")
    preamble = {"xhigh": "think hard", "medium": "", "low": "think briefly"}[effort]
    return f"{preamble}|tools={bool(tools)}"


def test_probe_learns_the_qwen38_vocabulary():
    profile = probe_effort_profile(_qwen38_render)
    assert profile.supported == frozenset({"xhigh", "medium", "low"})
    assert profile.default == "xhigh"
    assert profile.consumes_effort
    assert profile.validates  # rejections observed -> the vocabulary is real


def _dsv4_render(kwargs, tools):
    # Grades effort only in thinking mode (tools force it); asserts on unknown
    # values there; "low" renders the empty preamble, matching the default.
    effort = kwargs.get("reasoning_effort") or "low"
    if not tools:
        return "chat prompt"
    assert effort in ("low", "high", "max"), f"Invalid reasoning effort: {effort}"
    preamble = {"low": "", "high": "absolute maximum", "max": "beyond maximum"}[effort]
    return f"{preamble}|thinking"


def test_probe_learns_the_dsv4_vocabulary_through_the_tools_round():
    profile = probe_effort_profile(_dsv4_render)
    assert profile.supported == frozenset({"low", "high", "max"})
    assert profile.default == "low"
    assert profile.consumes_effort
    assert profile.validates


def test_probe_marks_an_ignoring_template_as_not_consuming():
    profile = probe_effort_profile(lambda kwargs, tools: f"same|tools={bool(tools)}")
    assert not profile.consumes_effort
    assert not profile.validates
    assert profile.default is None


def test_probe_marks_an_interpolating_template_as_not_validating():
    # Grades effort (renders differ) but rejects nothing: consumes without a
    # trustworthy vocabulary.
    profile = probe_effort_profile(lambda kwargs, tools: f"p|{kwargs.get('reasoning_effort')}")
    assert profile.consumes_effort
    assert not profile.validates
    # A non-validating grader that accepted the whole scale is capped to its
    # dialect's ladder -- the OpenAI triple for a plain effort grader.
    from freetoken.tokenizer.effort import effective_efforts  # noqa: E402

    assert effective_efforts(profile) == frozenset(OPENAI_EFFORT_TRIPLE)


def test_probe_skips_rounds_whose_baseline_fails():
    def render(kwargs, tools):
        if tools:  # template rejects tools outright -- round is uninformative
            raise RuntimeError("no tools supported")
        return _qwen38_render(kwargs, tools)

    profile = probe_effort_profile(render)
    assert profile.supported == frozenset({"xhigh", "medium", "low"})
    assert profile.consumes_effort


# --------------------------------------------------------------------------- #
# probe_thinking_profile: toggle behavior + default state
# --------------------------------------------------------------------------- #
def _thinking_render(baseline_state, kwargs, tools, adaptive=False):
    """A toggleable Qwen-style template: the off/on broadcasts render
    differently, and the no-kwarg baseline matches ``baseline_state``. When
    ``adaptive`` is False (the default) the template has no adaptive knob, so
    the adaptive probe is rejected -- only a minimax-style template admits it."""
    if "enable_thinking" in kwargs or "thinking_mode" in kwargs:
        if kwargs.get("thinking_mode") == "adaptive":
            if not adaptive:
                raise ValueError("template has no adaptive thinking mode")
            return "ADAPTIVE"
        if kwargs.get("enable_thinking") is False or kwargs.get("thinking_mode") == "disabled":
            return "OFF"
        if kwargs.get("enable_thinking") is True or kwargs.get("thinking_mode") == "enabled":
            return "ON"
    return baseline_state


def test_thinking_profile_toggleable_default_on():
    profile = probe_thinking_profile(lambda kw, t: _thinking_render("ON", kw, t), QWEN38)
    assert profile.toggleable is True
    assert profile.default_state == "on"
    assert profile.has_adaptive is False
    assert profile.efforts is QWEN38


def test_thinking_profile_default_off():
    profile = probe_thinking_profile(lambda kw, t: _thinking_render("OFF", kw, t), QWEN38)
    assert profile.toggleable is True
    assert profile.default_state == "off"


def test_thinking_profile_not_toggleable_when_broadcast_identical():
    # off and on render the same -> not a toggle.
    profile = probe_thinking_profile(lambda kw, t: "SAME", QWEN38)
    assert profile.toggleable is False
    assert profile.default_state == "on"
    assert profile.has_adaptive is False


def test_thinking_profile_rejecting_template_not_toggleable():
    def reject(kwargs, tools):
        raise RuntimeError("no thinking knobs here")

    profile = probe_thinking_profile(reject, QWEN38)
    assert profile.toggleable is False
    assert profile.has_adaptive is False
    assert profile.default_state == "on"


def test_thinking_profile_adaptive_third_state():
    # A minimax-m3-style template admits a third "adaptive" state distinct from
    # on/off; the no-kwarg baseline still matches "on".
    profile = probe_thinking_profile(lambda kw, t: _thinking_render("ON", kw, t, adaptive=True), QWEN38)
    assert profile.toggleable is True
    assert profile.has_adaptive is True
    assert profile.default_state == "on"


# --- scale table sanity ------------------------------------------------------
def test_scale_covers_the_probed_vocabulary():
    assert set(EFFORT_SCALE) == set(KNOWN_REASONING_EFFORTS)
    # xhigh and max intentionally share the top of the scale (0.99): max is an
    # opt-in alias, and quantization excludes "max" from rounding, so the tie
    # never makes nearest-gear selection ambiguous.
    assert EFFORT_SCALE["xhigh"] == EFFORT_SCALE["max"]
    assert OPENAI_EFFORT_TRIPLE == ("low", "medium", "high")
