"""Acknowledge Streamlit UI changes before the next browser interaction.

Full-script acknowledgements are distinct from a locally visible option/chip or
an idle attribute left over from the previous render. Never retry submissions.
"""
import re

from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.BackMsg_pb2 import BackMsg


class StreamlitSession:
    def __init__(self, page):
        self.page = page
        self.socket = None
        self.finished = 0
        page.on('websocket', self._connected)

    def _connected(self, socket):
        if '/_stcore/stream' in socket.url:
            self.socket = socket
            socket.on('framereceived', self._received)

    @staticmethod
    def _full_finish(payload):
        if not isinstance(payload, bytes):
            return False
        message = ForwardMsg.FromString(payload)
        return (message.WhichOneof('type') == 'script_finished' and message.script_finished in
                (ForwardMsg.FINISHED_SUCCESSFULLY, ForwardMsg.FINISHED_WITH_COMPILE_ERROR))

    def _received(self, payload):
        if self._full_finish(payload):
            self.finished += 1

    def wait_initial(self):
        assert self.socket is not None, 'Streamlit WebSocket was not established.'
        if not self.finished:
            payload = self.socket.wait_for_event('framereceived', predicate=self._full_finish, timeout=10000)
            assert ForwardMsg.FromString(payload).script_finished == ForwardMsg.FINISHED_SUCCESSFULLY
        self._rendered()

    def _rendered(self):
        from playwright.sync_api import expect
        expect(self.page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')
        # A wire acknowledgement precedes React's DOM/focus effects. Flush two
        # browser paint frames rather than sleeping an arbitrary duration.
        self.page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
        expect(self.page.get_by_test_id('stApp')).to_have_attribute('data-test-script-state','notRunning')

    def change(self, action, expected_value=None):
        assert self.socket is not None, 'Streamlit WebSocket was not established.'
        sent = False
        def request(payload):
            nonlocal sent
            if isinstance(payload, bytes):
                message = BackMsg.FromString(payload)
                if message.WhichOneof('type') == 'rerun_script':
                    widgets = message.rerun_script.widget_states.widgets
                    sent = sent or expected_value is None or any(
                        (list(w.string_array_value.data) == expected_value if isinstance(expected_value,list)
                         else w.WhichOneof('value') == 'string_value' and w.string_value == expected_value)
                        for w in widgets)
        self.socket.on('framesent', request)
        try:
            with self.socket.expect_event('framereceived',
                    predicate=lambda payload: sent and self._full_finish(payload), timeout=10000) as reply:
                action()
            assert ForwardMsg.FromString(reply.value).script_finished == ForwardMsg.FINISHED_SUCCESSFULLY
            self._rendered()
        finally:
            self.socket.remove_listener('framesent', request)


def workflow_state(page, expected):
    """Wait for a workflow boundary, then require the exact expected outcome.

    A failure/cancellation surfaces immediately, rather than consuming the whole
    completion allowance. The bounded allowance covers several sequential mock
    requests and UI renders, not just a single DOM transition.
    """
    from playwright.sync_api import expect
    status = page.get_by_text(re.compile(r'^Workflow status: '))
    states = 'completed|failed|cancelled|interrupted|waiting_approval'
    if expected == 'ready':
        states += '|ready'
    # Ready can also be the transient pre-claim state immediately after Start
    # or Continue; do not mistake it for the outcome of a running execution.
    expect(status).to_have_text(re.compile(r'^Workflow status: ('+states+r')$'), timeout=30000)
    expect(status).to_have_text('Workflow status: '+expected)
