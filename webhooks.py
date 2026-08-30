"""
Gesture Engine webhook client.

Responsibilities:
- Send gesture and hand-state events as HTTP POST JSON.
- Never block the WebRTC/CV processing thread.
- Keep webhook transport separate from prediction logic.
- Suppress duplicate events at the webhook-client level.

The decision about WHEN a state changes belongs to the Streamlit
video processor. This module only delivers the event it is given.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT = 2.0


class WebhookClient:
    """Asynchronous HTTP webhook client for Gesture Engine events."""

    def __init__(
        self,
        url: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.url = (url or "").strip()
        self.timeout = float(timeout)

        self._lock = threading.Lock()
        self._last_event: Optional[str] = None

    def set_url(self, url: str) -> None:
        """Update the endpoint used for future events."""
        with self._lock:
            new_url = (url or "").strip()

            # A changed/cleared endpoint starts with a clean event state.
            if new_url != self.url:
                self._last_event = None

            self.url = new_url

    def is_configured(self) -> bool:
        """Return True when a valid HTTP/HTTPS endpoint is configured."""
        with self._lock:
            return self.url.startswith(("http://", "https://"))

    def send_event(
        self,
        event: str,
        gesture: Optional[str] = None,
        confidence: float = 0.0,
        *,
        force: bool = False,
    ) -> bool:
        """
        Queue one webhook event.

        Returns True when a request was queued, otherwise False.

        `force` bypasses duplicate suppression. It is useful only when the
        caller explicitly needs to resend an identical event.
        """
        event = str(event)
        gesture_value = None if gesture is None else str(gesture)
        confidence_value = float(confidence or 0.0)

        with self._lock:
            url = self.url

            if not url.startswith(("http://", "https://")):
                return False

            event_key = f"{event}|{gesture_value}"

            if not force and event_key == self._last_event:
                return False

            self._last_event = event_key

        thread = threading.Thread(
            target=self._post_event,
            args=(url, event, gesture_value, confidence_value),
            daemon=True,
        )
        thread.start()

        return True

    def send_gesture(
        self,
        gesture: str,
        confidence: float,
    ) -> bool:
        """Backward-compatible helper for gesture_detected events."""
        return self.send_event(
            event="gesture_detected",
            gesture=gesture,
            confidence=confidence,
        )

    def send_no_hand(self) -> bool:
        """Send one hand_not_detected event until the state changes."""
        return self.send_event(
            event="hand_not_detected",
            gesture=None,
            confidence=0.0,
        )

    def reset(self) -> None:
        """Allow the current event to be sent again."""
        with self._lock:
            self._last_event = None

    def _post_event(
        self,
        url: str,
        event: str,
        gesture: Optional[str],
        confidence: float,
    ) -> None:
        payload: dict[str, Any] = {
            "event": event,
            "gesture": gesture,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        request = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Gesture-Engine-Webhook/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            # Webhook/network failure must never stop the CV/WebRTC stream.
            pass


def send_gesture_webhook(
    url: str,
    gesture: str,
    confidence: float,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Convenience function for one-off gesture delivery."""
    client = WebhookClient(url=url, timeout=timeout)
    return client.send_gesture(gesture, confidence)


def send_no_hand_webhook(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Convenience function for one-off no-hand delivery."""
    client = WebhookClient(url=url, timeout=timeout)
    return client.send_no_hand()
