"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            }
        )

        try:
            result = await self._send_callback(msg)
            self._sent_in_turn = True

            if isinstance(result, dict):
                parts = []

                if result.get("success") and result.get("text_sent"):
                    parts.append(f"✅ Message sent to {channel}:{chat_id}")

                sent = result.get("sent_files", [])
                if sent:
                    parts.append(f"✅ Attached {len(sent)} file(s)")

                failed = result.get("failed_files", [])
                if failed:
                    error_list = "\n".join([
                        f"  ❌ {path}: {reason}"
                        for path, reason in failed
                    ])
                    parts.append(f"❌ Failed to attach {len(failed)} file(s):\n{error_list}")

                error_msg = result.get("message", "")
                if error_msg and not result.get("success"):
                    parts.append(f"❌ Error: {error_msg}")

                return "\n".join(parts) if parts else "Error: Unknown status"
            else:
                media_count = len(media) if media else 0
                media_info = f" with {media_count} attachments" if media_count > 0 else ""
                return f"Message sent to {channel}:{chat_id}{media_info}"

        except Exception as e:
            return f"Error sending message: {str(e)}"
