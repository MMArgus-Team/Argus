"""Voice utterances must queue like typed prompts, and never render `user, user`.

Two things were wrong when a second utterance arrived while the first turn was
still running:

1. ``_submit_main`` painted the voice user bubble (``multimodal.asr_final``)
   *before* the busy gate, so a queued utterance appeared in the transcript
   immediately — producing two adjacent user messages with no assistant
   between them. History itself was fine (the FIFO is claimed under
   ``history_lock`` exactly like a typed prompt); the visible transcript was
   not, and ``user, user`` is the shape providers reject.
2. The turn tail releases ``running`` under the lock and only then re-acquires
   it to drain, so an utterance landing in that window took the direct path and
   jumped ahead of an older utterance still sitting in the FIFO.

These tests pin the queue/ordering contract at the gateway primitives the voice
path shares with the typed path.
"""

import threading
import unittest


class TestVoicePromptFifo(unittest.TestCase):
    def setUp(self):
        import tui_gateway.server as S
        self.S = S
        self._orig_submit = S._run_prompt_submit
        self._orig_emit = S._emit
        self.dispatched = []
        self.emitted = []

        def _fake_submit(rid, sid, session, text, **kwargs):
            self.dispatched.append({"rid": rid, "sid": sid, "text": text, **kwargs})
            with session["history_lock"]:
                session["running"] = False

        def _fake_emit(event, sid, payload=None, *a, **k):
            self.emitted.append({"event": event, "sid": sid, "payload": payload or {}})

        S._run_prompt_submit = _fake_submit
        S._emit = _fake_emit

    def tearDown(self):
        self.S._run_prompt_submit = self._orig_submit
        self.S._emit = self._orig_emit

    def _session(self, running=False):
        return {"history": [], "history_lock": threading.Lock(),
                "running": running, "_monitor_hook_running": False,
                "transport": None}

    def _asr_finals(self):
        return [e for e in self.emitted if e["event"] == "multimodal.asr_final"]

    def _enqueue_voice(self, session, text, seq, rid):
        return self.S._enqueue_prompt(
            session, text, session.get("transport"),
            user_originated=True, origin="voice_agent",
            metadata={"voice_task_seq": seq, "client_request_id": rid,
                      "voice_live_sid": "live-sid", "voice_turn_id": "turn-1"},
        )

    def test_queued_voice_bubble_is_painted_only_when_its_turn_starts(self):
        s = self._session(running=True)
        self._enqueue_voice(s, "第二句话", 2, "rid-2")

        # Still running → nothing dispatched, and crucially no bubble yet.
        self.assertFalse(self.S._drain_queued_prompt("d0", "sid", s))
        self.assertEqual(self._asr_finals(), [])

        # Turn A finishes → the queued utterance runs and paints its bubble now.
        with s["history_lock"]:
            s["running"] = False
        self.assertTrue(self.S._drain_queued_prompt("d1", "sid", s))

        finals = self._asr_finals()
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["payload"]["text"], "第二句话")
        self.assertEqual(finals[0]["payload"]["request_id"], "rid-2")
        self.assertEqual(finals[0]["payload"]["turn_id"], "turn-1")
        # Bubble is addressed to the live sid the utterance was captured on,
        # otherwise the frontend's isMine filter drops it.
        self.assertEqual(finals[0]["sid"], "live-sid")
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["text"], "第二句话")

    def test_bubble_precedes_its_own_dispatch(self):
        """The bubble must not appear after the answer starts streaming."""
        s = self._session(running=False)
        self._enqueue_voice(s, "一句话", 1, "rid-1")
        order = []
        _emit = self.S._emit

        def _tracking_emit(event, sid, payload=None, *a, **k):
            if event == "multimodal.asr_final":
                order.append("bubble")
            _emit(event, sid, payload, *a, **k)

        def _tracking_submit(rid, sid, session, text, **kwargs):
            order.append("dispatch")
            with session["history_lock"]:
                session["running"] = False

        self.S._emit = _tracking_emit
        self.S._run_prompt_submit = _tracking_submit
        self.assertTrue(self.S._drain_queued_prompt("d1", "sid", s))
        self.assertEqual(order, ["bubble", "dispatch"])

    def test_drain_preserves_spoken_order(self):
        s = self._session(running=True)
        for i, txt in enumerate(["第一句", "第二句", "第三句"], start=1):
            self._enqueue_voice(s, txt, i, f"rid-{i}")

        for _ in range(3):
            with s["history_lock"]:
                s["running"] = False
            self.S._drain_queued_prompt("d", "sid", s)

        self.assertEqual([d["text"] for d in self.dispatched],
                         ["第一句", "第二句", "第三句"])
        self.assertEqual([f["payload"]["text"] for f in self._asr_finals()],
                         ["第一句", "第二句", "第三句"])

    def test_non_voice_queued_prompt_paints_no_voice_bubble(self):
        """A typed queued prompt must not be given a voice bubble."""
        s = self._session(running=True)
        self.S._enqueue_prompt(s, "打字的问题", None,
                               user_originated=True, origin="user")
        with s["history_lock"]:
            s["running"] = False
        self.assertTrue(self.S._drain_queued_prompt("d1", "sid", s))
        self.assertEqual(self._asr_finals(), [])
        self.assertEqual(len(self.dispatched), 1)

    def test_direct_path_refused_while_anything_is_queued(self):
        """The ordering invariant the voice submit gate relies on.

        A fresh utterance may only run immediately when the session is idle AND
        the FIFO is empty; otherwise it would overtake older queued speech.
        """
        s = self._session(running=False)
        self._enqueue_voice(s, "较早的一句", 1, "rid-1")
        # running is False, but the queue is not empty → not eligible.
        eligible = not s.get("running") and not s.get("queued_prompts")
        self.assertFalse(eligible)

        # Drain it; now the session is genuinely free.
        self.S._drain_queued_prompt("d1", "sid", s)
        with s["history_lock"]:
            s["running"] = False
        self.assertTrue(not s.get("running") and not s.get("queued_prompts"))


class TestAttachmentOwnership(unittest.TestCase):
    """Only a prompt the human actually sent may consume composer attachments.

    The rule used to be "consume unless internal_origin is set", which spared
    only the two hidden hooks that happened to set that flag. Every other
    injected prompt — async-delegation completions, background process
    notifications, watch-pattern matches, goal follow-ups — silently swallowed
    the images the user had staged but not yet sent, and each new injected kind
    would have had to remember to opt out. The rule is now a positive
    requirement on ``user_originated``, whose default is already False.
    """

    def setUp(self):
        import tui_gateway.server as S
        self.S = S

    def _session(self, images):
        return {"attached_images": list(images),
                "history_lock": threading.Lock()}

    def test_a_real_user_submit_consumes_them(self):
        s = self._session(["/tmp/a.png", "/tmp/b.png"])
        got = self.S._claim_composer_attachments(s, user_originated=True)
        self.assertEqual(got, ["/tmp/a.png", "/tmp/b.png"])
        self.assertEqual(s["attached_images"], [], "consumed exactly once")

    def test_injected_prompt_neither_sees_nor_clears_them(self):
        """The staged image must survive to the user's next real submit."""
        s = self._session(["/tmp/screenshot.png"])
        got = self.S._claim_composer_attachments(s, user_originated=False)
        self.assertEqual(got, [], "an injected prompt must not receive the image")
        self.assertEqual(s["attached_images"], ["/tmp/screenshot.png"],
                         "and must not steal it from the composer either")

        # The user's real submit that follows still gets it.
        got2 = self.S._claim_composer_attachments(s, user_originated=True)
        self.assertEqual(got2, ["/tmp/screenshot.png"])

    def test_default_is_the_safe_one(self):
        """Injected callers are covered without opting in to anything."""
        import inspect
        sig = inspect.signature(self.S._run_prompt_submit)
        self.assertIs(sig.parameters["user_originated"].default, False)

    def test_missing_key_is_not_an_error(self):
        s = {"history_lock": threading.Lock()}
        self.assertEqual(
            self.S._claim_composer_attachments(s, user_originated=True), [])


class TestBusyInputIsConfigOwned(unittest.TestCase):
    """The config file owns interrupt-vs-queue for the TYPED path.

    Voice does not consult this at all (see TestVoiceNeverInterrupts): an
    utterance can never abort a running turn regardless of the setting.
    """

    def test_config_decides_and_default_is_interrupt(self):
        import tui_gateway.server as S
        orig = S._load_cfg
        try:
            for cfg, want in (
                ({"display": {}}, "interrupt"),
                ({}, "interrupt"),
                ({"display": {"busy_input_mode": "queue"}}, "queue"),
                ({"display": {"busy_input_mode": "steer"}}, "steer"),
                ({"display": {"busy_input_mode": "bogus"}}, "interrupt"),
            ):
                S._load_cfg = lambda c=cfg: c
                self.assertEqual(S._load_busy_input_mode(), want)
        finally:
            S._load_cfg = orig


if __name__ == "__main__":
    unittest.main()


class TestVoiceNeverInterrupts(unittest.TestCase):
    """Speech cannot be unsaid, so a second utterance must never abort the first.

    Two utterances are two separate requests; interrupting turn 1 to run turn 2
    throws away an answer the user already asked for. So the voice submit paths
    must enqueue directly and never route through _handle_busy_submit, whose
    default `interrupt` policy calls agent.interrupt(). display.busy_input_mode
    governs the TYPED path only.
    """

    def test_queue_only_never_interrupts_even_when_mode_is_interrupt(self):
        """The guarantee voice relies on, exercised for real.

        Voice enqueues directly, so it never reaches this function at all; the
        typed path reaches it and, with queue_only, must still not interrupt.
        """
        import tui_gateway.server as S

        interrupted = []

        class _Agent:
            def interrupt(self_inner):
                interrupted.append(True)

        orig_mode, orig_emit = S._load_busy_input_mode, S._emit
        try:
            S._load_busy_input_mode = lambda: "interrupt"
            S._emit = lambda *a, **k: None
            s = {"history": [], "history_lock": threading.Lock(),
                 "running": True, "agent": _Agent(),
                 "_monitor_hook_running": False, "transport": None}

            S._handle_busy_submit("r1", "sid", s, "Q2", None, queue_only=True)
            self.assertEqual(interrupted, [], "queue_only must not interrupt")

            S._handle_busy_submit("r2", "sid", s, "Q3", None, queue_only=False)
            self.assertEqual(len(interrupted), 1,
                             "the typed interrupt policy must still work")
            # Nothing was lost either way.
            self.assertEqual([q["text"] for q in s["queued_prompts"]],
                             ["Q2", "Q3"])
        finally:
            S._load_busy_input_mode, S._emit = orig_mode, orig_emit

    def test_voice_paths_queue_directly_and_check_the_backlog(self):
        """Architecture guard: both voice entry points bypass the policy fn.

        Asserted on source because reaching _submit_main / _dispatch_user_turn
        behaviourally needs a whole live VoiceAgent + ASR turn.
        """
        import inspect
        import tui_gateway.server as S

        src = inspect.getsource(S)
        for marker in ("def _submit_main(", "def _dispatch_user_turn("):
            self.assertIn(marker, src)
        # Neither voice path may call the interrupting policy helper.
        for fn_start in ("def _submit_main(", "def _dispatch_user_turn("):
            body = src[src.index(fn_start):][:6000]
            self.assertNotIn("_handle_busy_submit", body)
            self.assertIn("_enqueue_prompt", body)
            self.assertIn('session.get("queued_prompts")', body)


class TestTypedPromptOrdering(unittest.TestCase):
    """Q1 finishing must not let Q3 overtake an already-queued Q2.

    The turn tail sets running=False under history_lock and only re-acquires it
    afterwards to drain. A submit landing in that window used to see
    running=False and start immediately, jumping the queue.
    """

    def setUp(self):
        import tui_gateway.server as S
        self.S = S
        self._orig_submit = S._run_prompt_submit
        self._orig_emit = S._emit
        self.dispatched = []
        S._run_prompt_submit = self._fake_submit
        S._emit = lambda *a, **k: None

    def _fake_submit(self, rid, sid, session, text, **kwargs):
        self.dispatched.append(text)
        with session["history_lock"]:
            session["running"] = False

    def tearDown(self):
        self.S._run_prompt_submit = self._orig_submit
        self.S._emit = self._orig_emit

    def test_q3_landing_in_the_turn_boundary_gap_waits_behind_q2(self):
        s = {"history": [], "history_lock": threading.Lock(),
             "running": True, "_monitor_hook_running": False, "transport": None}

        # Q1 is running; Q2 arrives and queues.
        self.S._enqueue_prompt(s, "Q2", None, user_originated=True, origin="user")

        # Q1's turn tail releases the flag...
        with s["history_lock"]:
            s["running"] = False

        # ...and Q3 lands in the gap. The eligibility rule the submit gate uses
        # must reject the direct path because the FIFO is not empty.
        may_run_now = not s.get("running") and not s.get("queued_prompts")
        self.assertFalse(may_run_now, "Q3 must not start ahead of queued Q2")
        self.S._enqueue_prompt(s, "Q3", None, user_originated=True, origin="user")

        # Drain twice: spoken/typed order must be preserved.
        for _ in range(2):
            with s["history_lock"]:
                s["running"] = False
            self.S._drain_queued_prompt("d", "sid", s)

        self.assertEqual(self.dispatched, ["Q2", "Q3"])

    def test_typed_gate_consults_the_queue_not_just_running(self):
        import inspect
        fn = self.S._methods["prompt.submit"]
        self.assertIn("_queue_backlog", inspect.getsource(fn),
                      "prompt.submit must treat a non-empty FIFO as busy")
