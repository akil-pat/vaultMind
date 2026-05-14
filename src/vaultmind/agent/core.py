from __future__ import annotations

from collections.abc import Callable

import anthropic

from vaultmind.agent.tools import build_tools
from vaultmind.config import Settings
from vaultmind.ingest.embedder import Embedder
from vaultmind.ingest.store import VaultStore

SYSTEM_PROMPT = """\
You are vaultmind, a personal knowledge-base assistant. You have tools to \
search, list, and read the user's own notes (search_notes, list_notes, \
read_note).

Rules:
- If a question could plausibly be answered by something the user wrote \
  down, call search_notes before answering — do not guess from general \
  knowledge when their notes might have the actual answer.
- If search results are thin or ambiguous, try a differently-phrased query \
  or call read_note on the most promising result before giving up.
- When your answer draws on a note, cite it inline like [note.md > Heading] \
  so the user can find the source.
- If the notes don't cover the question, say so plainly and then answer \
  from general knowledge if you can, clearly marking that it isn't from \
  their notes.
- Be concise. This is a chat interface, not a report generator.
"""

ToolCallCallback = Callable[[str, dict], None]


class Agent:
    """Wraps the vector store, embedder, and Claude tool-use loop into a
    single conversational entry point. One Agent instance = one vault.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = VaultStore(settings.db_path)
        self.embedder = Embedder(settings.embedding_model)
        self.client = anthropic.Anthropic()
        self.tools = build_tools(self.store, self.embedder, settings.vault_path)

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        on_tool_call: ToolCallCallback | None = None,
    ) -> tuple[str, list[dict]]:
        """Ask a question, returning (answer_text, updated_history).

        Pass the returned history back in on the next call to continue the
        conversation. `on_tool_call` is invoked once per tool call the agent
        makes, before the tool executes — useful for a CLI "searching
        notes..." indicator.
        """
        messages = [*(history or []), {"role": "user", "content": question}]

        runner = self.client.beta.messages.tool_runner(
            model=self.settings.chat_model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=messages,
            output_config={"effort": self.settings.effort},
        )

        final_message = None
        for message in runner:
            final_message = message
            if on_tool_call is not None:
                for block in message.content:
                    if block.type == "tool_use":
                        on_tool_call(block.name, block.input)

        content = final_message.content if final_message is not None else []
        answer = "".join(block.text for block in content if block.type == "text")
        new_history = [*messages, {"role": "assistant", "content": content}]
        return answer, new_history
