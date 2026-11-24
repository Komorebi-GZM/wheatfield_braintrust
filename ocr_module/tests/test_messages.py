"""Prompt-shape tests.

v2 contract (training-aligned): exactly one user turn, exactly one real image
block, no few-shot example, no system message by default and no literal
"<image>" placeholder. See inference_protocol_v2.json -> prompt.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.prompts import INSTRUCTION, build_messages  # noqa: E402


class MessageTest(unittest.TestCase):
    def _blocks(self, messages):
        return [b for m in messages
                for b in (m["content"] if isinstance(m["content"], list) else [])]

    def test_one_real_image_and_no_phantom_example(self):
        messages = build_messages("page.jpg")
        blocks = self._blocks(messages)
        images = [b for b in blocks if b.get("type") == "image"]
        self.assertEqual(images, [{"type": "image", "url": "page.jpg"}])
        self.assertNotIn("<image>", str(messages))

    def test_single_user_turn_no_system_prompt_by_default(self):
        messages = build_messages("page.jpg")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    def test_text_block_is_newline_prefixed_instruction(self):
        # the training template put the instruction right after the image with a
        # leading newline; scripts/probe_protocol.py verified token-identical
        messages = build_messages("page.jpg")
        texts = [b for b in self._blocks(messages) if b.get("type") == "text"]
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text"], "\n" + INSTRUCTION)

    def test_system_prompt_only_on_request(self):
        messages = build_messages("page.jpg", use_system_prompt=True)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
