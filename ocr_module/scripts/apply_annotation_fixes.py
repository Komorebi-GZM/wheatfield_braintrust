"""Apply image-verified corrections to annotation_template.jsonl (v1 -> v2).

Every patch is anchored on an exact substring; a missing anchor aborts the run
so a stale/hand-edited file can never be silently half-patched.
"""
import json, os, sys, hashlib, datetime

SRC = 'datasets/handwriting_notes/annotation_template.jsonl'
BAK = 'datasets/handwriting_notes/annotation_template.v1.bak.jsonl'
LOG = 'outputs/annotation_fixes_v2.json'

PATCHES = [
    # (page basename, old, new, reason)
    ('IMG_20261104_200854.jpg',
     '主要参数：初始阻值 $R_0$、灵敏度系数 $K$、有效标距；机械滞回、疲劳寿命、绝缘电阻。',
     '主要参数：初始阻值 $R_0$、灵敏度系数 $K$、标称标距；机械滞回、疲劳寿命、绝缘电阻。',
     '原图该处写作“标称标距”，v1 记为教材术语“有效标距”，属术语替换。'),

    ('IMG_20261104_200858.jpg',
     '$$ \\sigma=\\frac{F}{A}\\approx 5.119\\times 10^7\\,\\mathrm{Pa} $$',
     '$$ S=\\frac{F}{A}\\approx 5.119\\times 10^7\\,\\mathrm{Pa} $$',
     '原图符号为 S（非 σ），v1 做了符号规范化。'),

    ('IMG_20261104_200858.jpg',
     '$$ \\varepsilon=\\frac{\\sigma}{E}\\approx 7.013\\times 10^{-4} $$',
     '$$ \\varepsilon=\\frac{S}{E}\\approx 7.013\\times 10^{-4} $$',
     '同上，原图为 ε=S/E。'),

    ('IMG_20261104_200858.jpg',
     '$$ \\sigma=\\frac{F}{A}\\approx 4.991\\times 10^7\\,\\mathrm{Pa} $$',
     '$$ S=\\frac{F}{A}\\approx 4.991\\times 10^7\\,\\mathrm{Pa} $$',
     '同上，3-6 的轴向应力原图写作 S。'),

    ('IMG_20261104_235916.jpg',
     '| 字寄存器 | 高字节 | 低字节 | 名称 |\n|---|---|---|---|\n| AX | AH | AL | 累加器 |\n| BX | BH | BL | 基数寄存器 |\n| CX | CH | CL | 计数寄存器 |\n| DX | DH | DL | 数据寄存器 |',
     'AX AH AL 累加器\n\nBX BH BL 基数寄存器\n\nCX CH CL 计数寄存器\n\nDX DH DL 数据寄存器',
     '原图是四列竖排罗列，没有表头行；v1 自行添加了“字寄存器/高字节/低字节/名称”表头。'),

    ('IMG_20261104_235916.jpg',
     'ALU：运算；进行各种寻址方式的计算，得出 16 位偏移地址。',
     'ALU：运算、逻辑；按指令寻址方式计算出 16 位偏移地址。',
     '原图 ALU 行含“逻辑”，v1 漏抄并改写了措辞。'),

    ('IMG_20261104_235922.jpg',
     '1. 段寄存器（最高 16 位，低 4 位为 0）：\n   - CS：代码段寄存器。\n   - DS：数据段寄存器。\n   - ES：扩展段寄存器。\n   - SS：堆栈段寄存器。\n2. 16 位指令指针寄存器 IP。\n3. 20 位地址加法器。\n4. 6B 指令队列缓冲器。\n5. 16 位内部寄存器。\n6. 总线逻辑控制器。',
     '1. CS：代码段寄存器（最高 16 位，低 4 位为 0）。\n2. DS：数据段寄存器。\n3. ES：扩展段寄存器。\n4. SS：堆栈段寄存器。\n5. 16 位指令指针寄存器 IP。\n6. 20 位地址加法器。\n7. 6B 指令队列缓冲器。\n8. 16 位内部寄存器。\n9. 总线逻辑控制器。',
     '原图 CS/DS/ES/SS 是 1~4 的独立条目，没有“段寄存器”这一上级；v1 自造了层级并因此错位编号。'),

    ('IMG_20261104_235946.jpg',
     '### 传送指令\n\n`MOV 目的操作数, 源`\n\n1. IP 不作目的。\n2. 立即数作源。\n3. 源、目的类型一致，字或字节。\n4. 注意大小。\n5. 不允许段寄存器间传送（可对外，不对内）。\n6. 立即数不作目的，不传段寄存器。\n7. `MOV BUF1, BUF2`：不同时对存储单元操作，6、7 可借助 AX。\n8. 段与段之间不传送。',
     '### 传送指令\n\n`MOV 目的操作数，源`（用字或字节）\n\n1. IP 不作目的。\n2. 立即数不作目的。\n3. 源、目的类型一致。\n4. 注意大小。\n5. 不能给段寄存器（CS）赋值（同内，不对内）。\n6. MOV 不改变源内容，不影响标志位。\n6. 立即数不作目的，不传段寄存器。\n7. `MOV BUF1, BUF2`；不同存储单元操作，6、7 可借助 AX。\n8. 段与段之间不传送。',
     'v1 第 2 条误作“立即数作源”（原图为“不作目的”），第 5 条把“不能给 CS 赋值”改写成“不允许段寄存器间传送”，并漏抄“不改变源内容、不影响标志位”；原图第 6 条重复编号两次，此处照录。'),

    ('IMG_20261105_000028.jpg',
     '$A_1,A_0$；$\\overline{Y_i}$ 取 $A_4,A_3,A_2$。',
     '$A_1,A_0$；$Y_i$ 取 $A_4,A_3,A_2$。\n\nA₁ A₀{300 uA／10□u B／40 n C',
     '原图红字“A₁A₀{300 uA/10□u B/40 n C”在 v1 中整段漏抄；同时“取 A4,A3,A2”前是插入符号而非上划线，去掉多余 \\overline。单元字符按原图保留□。'),

    ('IMG_20261105_000037.jpg',
     '$D_7=0$；$D_6D_5D_4$ 任意值',
     '$D_7=0$；$D_6D_5$ 任意值',
     '原图为两位 D6D5 任意值；v1 写成三位 D6D5D4（8255 控制字的习惯写法），属按知识点补写。'),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = [json.loads(l) for l in open(SRC, encoding='utf-8') if l.strip()]
    by = {os.path.basename(r['image']): r for r in rows}
    if not os.path.exists(BAK):
        with open(BAK, 'w', encoding='utf-8', newline='\n') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    before = sha(SRC)
    applied = []
    for page, old, new, reason in PATCHES:
        row = by.get(page)
        if row is None:
            sys.exit(f'anchor page missing: {page}')
        if row['markdown'].count(old) != 1:
            sys.exit(f'anchor not unique/found on {page}: {old[:40]!r} count={row["markdown"].count(old)}')
        row['markdown'] = row['markdown'].replace(old, new)
        applied.append({'page': page, 'reason': reason, 'old': old, 'new': new})
    with open(SRC, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    out = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'file': SRC,
        'backup_v1': BAK,
        'sha256_before': before,
        'sha256_after': sha(SRC),
        'n_pages': len(rows),
        'n_patches': len(applied),
        'pages_touched': sorted({a['page'] for a in applied}),
        'patches': applied,
    }
    os.makedirs('outputs', exist_ok=True)
    with open(LOG, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'patches'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
