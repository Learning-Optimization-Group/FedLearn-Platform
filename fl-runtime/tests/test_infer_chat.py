import os, sys
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SC)
import infer  # noqa

def test_render_chat_single_turn_matches_legacy():
    assert infer.render_chat([], "hi") == "### Instruction:\nhi\n### Response:\n"
    assert infer.render_chat(None, "hi") == "### Instruction:\nhi\n### Response:\n"

def test_render_chat_multi_turn():
    h = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert infer.render_chat(h, "c") == "### Instruction:\na\n### Response:\nb\n### Instruction:\nc\n### Response:\n"
