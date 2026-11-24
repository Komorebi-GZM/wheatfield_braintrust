"""CPU-only probe: compare Swift *training* rendering vs repo *inference* rendering.

No model weights are loaded, so this cannot be mistaken for a real inference run.

Variants
  A  Swift template (mode=train, max_pixels=1024000)  -> what training actually saw
  B  current repo inference messages (system + list content, max_pixels applied)
  C  training-aligned inference (no system, "<image>\\n" string, max_pixels applied)

Writes outputs/protocol/probe_<page>.json
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from swift.model import get_model_processor
from swift.template import get_template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.prompts import SYSTEM_PROMPT, USER_INSTRUCTION  # noqa: E402

INSTRUCTION_TEXT = USER_INSTRUCTION.replace('<image>\n', '')
PAD = '<|image_pad|>'


def flat(ids):
    if ids and isinstance(ids[0], list):
        return list(ids[0])
    return list(ids)


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return -1 if len(x) == len(y) else min(len(x), len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default='IMG_20261104_200903.jpg')
    ap.add_argument('--max-pixels', type=int, default=1024000)
    ap.add_argument('--out', default='outputs/protocol')
    args = ap.parse_args()

    img = str(Path('datasets/handwriting_notes/images') / args.page)
    model_path = snapshot_download('SeerRay-Lab/Xiaomi-OCR-0', local_files_only=True)
    _, processor = get_model_processor(model_path, model_type='qwen3_5', load_model=False)
    tok = processor.tokenizer
    pad_id = tok.convert_tokens_to_ids(PAD)

    # A: Swift training rendering (applies its own max_pixels)
    tpl = get_template(processor, template_type='qwen3_5', max_length=8192, max_pixels=args.max_pixels)
    tpl.set_mode('train')
    enc_a = tpl.encode({'messages': [{'role': 'user', 'content': '<image>\n' + INSTRUCTION_TEXT},
                                     {'role': 'assistant', 'content': '占位'}],
                        'images': [img]})
    a_ids = flat(enc_a['input_ids'])
    a_grid = enc_a['image_grid_thw'].tolist()

    # Apply the same pixel budget to the HF processor used at inference time.
    base_size = dict(processor.image_processor.size)
    processor.image_processor.size = {'shortest_edge': base_size['shortest_edge'],
                                      'longest_edge': args.max_pixels}
    pil = Image.open(img).convert('RGB')

    # B: current repo inference messages
    enc_b = processor.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': [{'type': 'image', 'url': img},
                                      {'type': 'text', 'text': INSTRUCTION_TEXT}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    b_ids = flat(enc_b['input_ids'].tolist())
    b_grid = enc_b['image_grid_thw'].tolist()

    # C: training-aligned inference = no system message + newline after the image,
    #    image supplied as a content block so the chat template emits the vision tokens.
    enc_c = processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image', 'url': img},
                                      {'type': 'text', 'text': '\n' + INSTRUCTION_TEXT}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    c_ids = flat(enc_c['input_ids'].tolist())
    c_grid = enc_c['image_grid_thw'].tolist()

    def summarise(ids, grid):
        txt = tok.decode(ids, skip_special_tokens=False)
        compact = txt.replace(PAD, '')
        return {'n_tokens': len(ids), 'grid': grid, 'n_image_pad': ids.count(pad_id),
                'tail_ids': ids[-6:],
                'tail_text_compact': compact[-160:],
                'head_text_compact': compact[:120]}

    # --- assistant-prefix / stop conditions ---------------------------------
    # The generation prompt must reproduce the same assistant header Swift used,
    # including the *empty think block* the qwen3_5 template emits. Note the
    # header is "assistant\n<think>\n\n</think>\n\n", not just "assistant\n".
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    a_tail3 = a_ids[-3:]
    im_start = tok.convert_tokens_to_ids('<|im_start|>')
    # last assistant header inside C = everything from the last <|im_start|> on
    last_hdr = len(c_ids) - 1 - c_ids[::-1].index(im_start)
    c_header_ids = c_ids[last_hdr:]
    c_header_text = tok.decode(c_header_ids, skip_special_tokens=False)
    a_header_ids = a_ids[last_hdr:last_hdr + len(c_header_ids)]
    thinking = {
        'assistant_header_text': c_header_text,
        'assistant_header_ids': c_header_ids,
        'A_same_header': a_header_ids == c_header_ids,
        'header_contains_empty_think': '<think>' in c_header_text and '</think>' in c_header_text,
        'note': '空 <think>\\n\\n</think> 块由 qwen3_5 模板生成，训练与推理两侧一致即视为非思考模式',
    }

    report = {
        'page': args.page,
        'page_sha256': hashlib.sha256(Path(img).read_bytes()).hexdigest(),
        'model_dir': model_path,
        'max_pixels': args.max_pixels,
        'original_size_wh': list(pil.size),
        'tokens': {'image_pad': pad_id, 'im_end': im_end,
                   'endoftext': tok.convert_tokens_to_ids('<|endoftext|>'),
                   'vision_start': tok.convert_tokens_to_ids('<|vision_start|>'),
                   'eos_token': tok.eos_token, 'pad_token': tok.pad_token},
        'A_swift_train': summarise(a_ids, a_grid),
        'B_repo_inference': summarise(b_ids, b_grid),
        'C_aligned': summarise(c_ids, c_grid),
        'assistant_prefix_check': {
            'C_ends_with_A_header': c_header_ids == a_header_ids,
            'C_tail_text': tok.decode(c_ids[-8:], skip_special_tokens=False).replace(PAD, ''),
            'answer_starts_right_after_prompt': a_ids[len(c_ids):len(c_ids) + 2],
        },
        'stop_token_check': {
            'A_tail4_ids': a_ids[-4:],
            'A_tail4_text': tok.decode(a_ids[-4:], skip_special_tokens=False),
            'A_second_last_is_im_end': a_ids[-2] == im_end,
            'A_last_is_newline': tok.decode([a_ids[-1]]) == '\n',
            'im_end_equals_eos_token_id': im_end == tok.convert_tokens_to_ids(tok.eos_token),
            'note': '监督序列以 <|im_end|>\\n 结束；生成时以 im_end(248046) 为 EOS 提前终止，尾随换行不参与生成',
        },
        'thinking_check': thinking,
        'diff': {'A_vs_B_first': first_diff(a_ids, b_ids),
                 'A_vs_B_len_delta': len(b_ids) - len(a_ids),
                 'A_prefix_vs_C_first': first_diff(a_ids[:len(c_ids)], c_ids),
                 'A_prefix_vs_C_len_delta': len(c_ids) - len(a_ids),
                 'A_next_tokens_after_prompt': a_ids[len(c_ids):len(c_ids) + 6],
                 'A_next_tokens_text': tok.decode(a_ids[len(c_ids):len(c_ids) + 6], skip_special_tokens=False),
                 'prompt_identical': a_ids[:len(c_ids)] == c_ids},
    }
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"probe_{Path(args.page).stem}.json")
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('written', p)


if __name__ == '__main__':
    main()
