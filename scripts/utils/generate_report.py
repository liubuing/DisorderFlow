"""鐢熸垚 BFN V5 璁粌椤圭洰璇︾粏鎶ュ憡锛?026-05-16/17 椤圭洰鏃ワ級"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os
from datetime import datetime

OUTPUT = os.path.join(os.path.dirname(__file__), "reports", "BFN_V5_璁粌鎶ュ憡_2026_05_17.docx")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

doc = Document()

# 鏍峰紡璁剧疆
style = doc.styles['Normal']
font = style.font
font.name = 'Microsoft YaHei'
font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.color.rgb = RGBColor(0x1A, 0x56, 0x8E)
    hs.element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
    return table

def add_para(text, bold=False, size=10):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p

# ============================================================
# 灏侀潰
# ============================================================
title = doc.add_heading('BFN 缃俊搴﹀ご閮?V5 璁粌鎶ュ憡', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('椤圭洰鏃ワ細2026-05-17锛?026-05-16 01:30 ~ 2026-05-17 01:30锛?)
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(f'鎶ュ憡鐢熸垚鏃堕棿锛歿datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
run.font.size = Pt(9)
run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

doc.add_paragraph()

# ============================================================
# 1. 鎬昏
# ============================================================
doc.add_heading('1. 椤圭洰鎬昏', level=1)
doc.add_paragraph(
    '鏈姤鍛婅褰曚簡 BFN 缃俊搴﹀ご閮?V5 璁粌椤圭洰鐨勫畬鏁存祦绋嬶細V4 鍩虹嚎璇勪及 鈫?鐤剧梾鏁版嵁闆?v2 鎵╁睍 鈫?'
    '缃俊搴︽帺鐮佸寮虹畻娉曞疄鐜?鈫?鏁版嵁闆嗗悎骞?鈫?V5 闃舵涓€/闃舵浜岃缁冦€傛牳蹇冪洰鏍囨湁涓や釜锛?
)
for item in [
    '灏嗚璁℃椂鐨?ipTM 棰勬祴浠?~0.695 鎻愬崌鍒?0.75 浠ヤ笂锛堢悊鎯?0.8+锛夛紝涓斿繀椤荤鍚堢湡瀹炴姌鍙犵墿鐞嗚寰?,
    '灏嗙柧鐥呰泲鐧芥暟鎹泦浠?275 鏉℃墿灞曡嚦 500+ 鏉★紝澧炲己 ipTM 棰勬祴鐨勫鏍锋€у拰瑕嗙洊鑼冨洿',
]:
    doc.add_paragraph(item, style='List Bullet')

doc.add_paragraph(
    '鍏抽敭鎴愭灉锛歏4 鍦?126 铔嬬櫧楠岃瘉闆嗕笂杈惧埌 pLDDT Pearson r=0.839銆乮pTM r=0.893銆?
    'V5 闃舵涓€锛堟棫鏁版嵁闆?629 鏉★級鏈€浣抽獙璇佹崯澶?0.0316锛堢 4700 娆¤凯浠ｏ級銆?
    'V5 闃舵浜岋紙鎵╁厖鏁版嵁闆?1149 鏉★紝璁粌 919/楠岃瘉 230锛夋鍦ㄨ缁冧腑锛岄獙璇佹崯澶变粠 0.0607 绋虫涓嬮檷鑷?0.0451锛堢 1100 娆¤凯浠ｏ級锛岄檷骞?25.7%銆?
)

# ============================================================
# 2. V4 鍩虹嚎
# ============================================================
doc.add_heading('2. V4 鍩虹嚎璁粌缁撴灉', level=1)

doc.add_heading('2.1 璁粌閰嶇疆', level=2)
add_table(
    ['鍙傛暟', '閰嶇疆鍊?],
    [
        ['鍩虹妫€鏌ョ偣', 'logs/bfn_confidence_combined_2026_05_16__14_11_01/checkpoints/best.pt'],
        ['鏁版嵁闆?, 'v3 (474璁粌+121楠岃瘉) + IDP v2 (41鏉? = 503璁粌 / 126楠岃瘉'],
        ['鎹熷け鏉冮噸', 'pLDDT=1.0, ipTM=1.0, PAE=0.3'],
        ['璁粌鍥炴敹娆℃暟', '2'],
        ['鍐荤粨楠ㄥ共缃戠粶', 'True锛堜粎璁粌缃俊搴﹀ご閮?+ 鏈€鍚?灞傜紪鐮佸櫒锛?],
        ['鍙缁冨弬鏁伴噺', '2,930,139 / 10,237,462 (28.6%)'],
        ['pLDDT 澶撮儴鏋舵瀯', 'Linear(256鈫?28)鈫扲eLU鈫扡inear(128鈫?2)鈫扲eLU鈫扡inear(32鈫?)'],
        ['鏈€澶ц凯浠ｆ鏁?, '30,000'],
        ['鎵规澶у皬', '1'],
        ['浼樺寲鍣?, 'Adam (lr=1e-4, 尾1=0.9, 尾2=0.999)'],
        ['瀛︿範鐜囪皟搴﹀櫒', 'Plateau (factor=0.5, patience=30, min_lr=1e-6)'],
        ['姊害瑁佸壀', 'max_norm=1.0'],
        ['璁粌寮€濮嬫椂闂?, '2026-05-16 23:04:14'],
        ['璁粌缁撴潫鏃堕棿', '2026-05-17 00:40:50'],
        ['鎬昏€楁椂', '绾?96 鍒嗛挓'],
    ]
)

doc.add_heading('2.2 楠岃瘉鎹熷け鍏抽敭鑺傜偣', level=2)
add_table(
    ['杩唬', '楠岃瘉鎹熷け', 'pLDDT', 'ipTM', 'PAE', '鏀硅繘骞呭害'],
    [
        ['100', '0.0760', '0.0430', '0.0169', '0.0535', '鍒濆鍊?],
        ['500', '0.0469', '0.0167', '0.0172', '0.0431', '-0.0291'],
        ['1000', '0.0447', '0.0159', '0.0164', '0.0412', '-0.0313'],
        ['2700', '0.0363', '0.0146', '0.0115', '0.0341', '-0.0397'],
        ['5400', '0.0335', '0.0134', '0.0113', '0.0293', '-0.0425'],
        ['8100', '0.0326', '0.0124', '0.0113', '0.0294', '-0.0434'],
        ['12800', '0.0308', '0.0120', '0.0106', '0.0271', '-0.0452'],
        ['14800', '0.0291', '0.0120', '0.0090', '0.0266', '-0.0469'],
        ['26600锛堟渶浣筹級', '0.0283', '0.0113', '0.0092', '0.0262', '-0.0477'],
        ['30000锛堟渶缁堬級', '0.0295', '0.0120', '0.0096', '0.0265', '-0.0465'],
    ]
)

doc.add_heading('2.3 126铔嬬櫧楠岃瘉闆嗚瘎浼扮粨鏋?, level=2)
add_table(
    ['鎸囨爣', 'V3', 'V4', '鍙樺寲'],
    [
        ['pLDDT Pearson r', '0.652', '0.839', '+0.187 (+28.7%)'],
        ['pLDDT Spearman 蟻', '鈥?, '0.760', '鈥?],
        ['pLDDT MAE', '鈥?, '0.073', '鈥?],
        ['ipTM Pearson r', '0.824', '0.893', '+0.069 (+8.4%)'],
        ['ipTM Spearman 蟻', '鈥?, '0.897', '鈥?],
        ['ipTM MAE', '鈥?, '0.079', '鈥?],
        ['姣忚泲鐧?pLDDT 鍧囧€?r', '鈥?, '0.716', '鈥?],
        ['pLDDT r>0.8 鐨勮泲鐧芥暟', '鈥?, '74/126 (58.7%)', '鈥?],
    ]
)

doc.add_heading('2.4 璁捐鏃?ipTM 鍋忎綆闂', level=2)
doc.add_paragraph(
    '灏界楠岃瘉闆嗘寚鏍囪〃鐜颁紭绉€锛坕pTM r=0.893锛夛紝浣嗚璁℃椂 ipTM 濮嬬粓鍗″湪 ~0.695锛堢洰鏍?0.75+锛夈€?
    '鏍规湰鍘熷洜锛氳缁?璁捐鍒嗗竷涓嶅尮閰嶃€傝缁冮樁娈电紪鐮佸櫒鍙瀹屾暣缁撴瀯锛坓enerate_flag=0锛夛紝'
    '浣嗚璁￠樁娈电紪鐮佸櫒浠呰兘鐪嬪埌涓婁笅鏂囨畫鍩猴紙generate_flag=1锛夛紝鐢熸垚鍖哄煙琚～鍏呬负鎵╁睍铻烘棆楠ㄦ灦銆?
    'ipTM 澶撮噰鐢ㄥ叏灞€骞冲潎姹犲寲锛岀敓鎴愬尯鍩熺己澶辩殑缁撴瀯淇℃伅浼氱█閲婃暣涓泲鐧界殑棰勬祴淇″彿锛?
    '瀵艰嚧 ipTM 鍑犱箮鎭掑畾杈撳嚭 ~0.695锛屾棤娉曞弽鏄犺璁¤川閲忕殑鐪熷疄宸紓銆?
)

# ============================================================
# 3. 鐤剧梾鏁版嵁闆?v2
# ============================================================
doc.add_heading('3. 鐤剧梾鏁版嵁闆?v2 鎵╁睍', level=1)

doc.add_heading('3.1 鏋勫缓鑴氭湰锛歜uild_disease_dataset_v2.py锛堟柊寤猴級', level=2)
doc.add_paragraph('鐩告瘮 v1 鐗堟湰锛坆uild_disease_dataset.py锛?75 鏉★紝10 涓被鍒級鐨勪富瑕佹敼杩涳細')
for item in [
    '鏌ヨ鏁伴噺浠?10 涓墿灞曞埌 19 涓細10 涓墿灞曞師鏈夌被鍒?+ 6 涓柊澧炵柧鐥呯被鍒?+ 3 涓叧閿瘝鏌ヨ',
    '闀垮害鑼冨洿鎵╁ぇ锛氫粠 50-250aa 鎵╁睍鑷?50-400aa锛堣鐩栨縺閰躲€佸彈浣撶瓑澶ц泲鐧斤級',
    'UniProt 鍒嗛〉锛氭瘡娆℃煡璇㈡渶澶氳幏鍙?200 鏉★紙鍘熶负 40 鏉★級锛屼娇鐢ㄦ父鏍囧垎椤垫満鍒?,
    '鏂板 6 涓柧鐥呯被鍒細鑷韩鍏嶇柅(41)銆佺毊鑲ょ梾(41)銆佸唴鍒嗘硨(44)銆佽儍鑲犻亾(27)銆佺嚎绮掍綋(40)銆佺缁忓彂鑲?36)',
    '鏂板 3 涓叧閿瘝鏌ヨ锛歞isease_mutation (KW-0225)銆乷ncogene (KW-0590)銆乧ancer_driver',
    '姣忔潯鏁版嵁鏍囪 source="Disease_v2_{category}" 鍜?is_idp=False',
    '姣忓鐞?50 鏉¤嚜鍔ㄤ繚瀛樹竴娆★紝闃叉宕╂簝涓㈠け杩涘害',
]:
    doc.add_paragraph(item, style='List Bullet')

doc.add_heading('3.2 鏋勫缓缁撴灉', level=2)
add_table(
    ['绫诲埆', '鏉＄洰鏁?],
    [
        ['Disease_v2_liver_disease锛堣倽鐥咃級', '31'],
        ['Disease_v2_kidney_disease锛堣偩鐥咃級', '35'],
        ['Disease_v2_heart_disease锛堝績鑴忕梾锛?, '23'],
        ['Disease_v2_lung_disease锛堣偤鐥咃級', '33'],
        ['Disease_v2_neuro_disease锛堢缁忕柧鐥咃級', '28'],
        ['Disease_v2_cancer锛堢檶鐥囷級', '31'],
        ['Disease_v2_metabolic_disease锛堜唬璋㈢梾锛?, '26'],
        ['Disease_v2_blood_disease锛堣娑茬梾锛?, '31'],
        ['Disease_v2_muscle_disease锛堣倢鑲夌梾锛?, '28'],
        ['Disease_v2_eye_disease锛堢溂鐥咃級', '24'],
        ['Disease_v2_autoimmune_disease锛堣嚜韬厤鐤級', '41'],
        ['Disease_v2_skin_disease锛堢毊鑲ょ梾锛?, '41'],
        ['Disease_v2_endocrine_disease锛堝唴鍒嗘硨锛?, '44'],
        ['Disease_v2_gi_disease锛堣儍鑲犻亾锛?, '27'],
        ['Disease_v2_mitochondrial_disease锛堢嚎绮掍綋锛?, '40'],
        ['Disease_v2_neurodevelopmental_disease锛堢缁忓彂鑲诧級', '36'],
        ['Disease_v2_oncogene锛堢檶鍩哄洜锛?, '24'],
        ['Disease_v2_cancer_driver锛堢檶鐥囬┍鍔ㄥ熀鍥狅級', '2'],
        ['鎬昏', '545'],
    ]
)
doc.add_paragraph(
    '鏈€缁?LMDB锛?45 鏉★紝927 MB銆傛垚鍔?545/546锛? 鏉″け璐ワ級銆?
    '鏁版嵁闆嗕腑鍖呭惈澶ч噺浣?pTM 鏍锋湰锛? 0.3锛夛紝涓虹疆淇″害澶撮儴鎻愪緵浜嗘鍓嶇己澶辩殑浣?ipTM 璁粌淇″彿銆?
)

# ============================================================
# 4. 浠ｇ爜淇敼
# ============================================================
doc.add_heading('4. 浠ｇ爜淇敼锛氱疆淇″害鎺╃爜澧炲己', level=1)

doc.add_heading('4.1 淇敼鏂囦欢', level=2)
doc.add_paragraph(
    'disorderflow/modules/bfn/core.py锛岀 85-130 琛岄檮杩戯細鏂板缃俊搴︽帺鐮佸寮洪€昏緫锛?
    '鐢ㄤ簬寮ュ悎璁粌/璁捐鍒嗗竷宸窛銆傝繖鏄?V5 鐗堟湰鐨勬牳蹇冪畻娉曞垱鏂般€?
)

doc.add_heading('4.2 绠楁硶鍘熺悊', level=2)
doc.add_paragraph(
    '璁粌鏃朵互 40% 鐨勬鐜囷紙mask_aug_prob=0.4锛夛紝瀵规壒娆′腑 20%-100% 鐨勬畫鍩洪殢鏈鸿缃?generate_flag=1锛?
    '骞跺悓鏃跺皢杩欎簺娈嬪熀鐨?pair_feat 褰掗浂銆傝繖浣垮緱缂栫爜鍣ㄥ鎺╃爜娈嬪熀鏃犳硶鑾峰彇浠讳綍缁撴瀯淇℃伅锛?
    'ipTM 澶撮儴蹇呴』浠庨儴鍒嗕俊鎭腑棰勬祴鍏ㄨ泲鐧界殑 ipTM锛屼粠鑰屾ā鎷熻璁℃椂鐨勭湡瀹炴潯浠躲€?
)

doc.add_paragraph('绠楁硶姝ラ锛?)
for s in [
    '姝ラ1锛氳绠?N_batch = x_seq.size(0)锛岀‘淇濆彉閲忓湪鎺╃爜寰幆鍓嶅畾涔夛紙淇浜?UnboundLocalError 閿欒锛?,
    '姝ラ2锛氫互姒傜巼 mask_aug_prob (0.4) 瑙﹀彂澧炲己锛氫粠 Uniform(0.2, 1.0) 闅忔満閲囨牱澧炲己姣斾緥',
    '姝ラ3锛氬姣忎釜鎵规鍏冪礌锛屾壘鍑虹湡瀹炴畫鍩猴紙闈炲～鍏呫€佹湭鎺╃爜锛夛細real_mask = mask_res & ~mask_gen',
    '姝ラ4锛氶殢鏈烘墦涔辩湡瀹炴畫鍩虹储寮曪紝閫夊彇 ratio 脳 len(real_idx) 涓畫鍩猴紝璁剧疆 mask_gen[b, chosen] = True',
    '姝ラ5锛氬皢琚寮烘畫鍩虹殑 pair_feat 褰掗浂锛堣鍒椾袱涓柟鍚戯級锛屾秷闄ょ紪鐮佸櫒浠庨厤瀵圭壒寰佷腑鎺ㄦ柇缁撴瀯鐨勫彲鑳芥€?,
    '姝ラ6锛氭帴鏀跺櫒浠庨儴鍒嗕俊鎭腑棰勬祴锛屼絾鎹熷け鍑芥暟浠嶄互瀹屾暣 AF2 鐪熷疄鍊间负鐩爣杩涜璁＄畻',
]:
    doc.add_paragraph(s, style='List Bullet')

doc.add_paragraph(
    '鍏抽敭璁捐锛歱air_feat 褰掗浂鑷冲叧閲嶈銆傚鏋滀笉褰掗浂锛屽嵆浣?generate_flag=1锛岀紪鐮佸櫒浠嶅彲閫氳繃閰嶅鐗瑰緛鎺ㄦ柇缁撴瀯鍏崇郴锛?
    '閮ㄥ垎鍓婂急澧炲己鏁堟灉銆傚綊闆跺悗妯″瀷琚揩渚濊禆搴忓垪涓婁笅鏂囧拰鍏ㄥ眬妯″紡杩涜棰勬祴鈥斺€旇繖姝ｆ槸璁捐鏃剁殑鐪熷疄鍦烘櫙銆?
)

doc.add_heading('4.3 閿欒淇锛歎nboundLocalError', level=2)
doc.add_paragraph(
    '绗竴娆?V5 璁粌灏濊瘯宕╂簝锛岄敊璇俊鎭細"UnboundLocalError: cannot access local variable \'N\' '
    'where it is not associated with a value"锛坈ore.py 绗?92 琛岋級銆?
    '鎺╃爜澧炲己寰幆涓娇鐢ㄤ簡鍙橀噺 N锛屼絾 N 鍦ㄥ師浠ｇ爜涓畾涔変簬绗?102 琛岋紙N = x_seq.size(0)锛夛紝浣跨敤鏃跺皻鏈畾涔夈€?
    '淇鏂规锛氬湪鎺╃爜澧炲己浠ｇ爜鍧椾箣鍓嶆坊鍔?N_batch = x_seq.size(0)锛堢 86 琛岋級锛屽苟鍦ㄥ惊鐜腑浣跨敤 N_batch銆?
)

# ============================================================
# 5. V5 閰嶇疆
# ============================================================
doc.add_heading('5. V5 璁粌閰嶇疆', level=1)

doc.add_heading('5.1 閰嶇疆鏂囦欢锛歝onfigs/train/bfn_confidence_combined_v5.yml锛堟柊寤猴級', level=2)
add_table(
    ['閰嶇疆鑺?, '鍙傛暟', 'V4 鍊?, 'V5 鍊?, '鍙樻洿鍘熷洜'],
    [
        ['model.loss_weight', 'plddt', '1.0', '1.0', '涓嶅彉'],
        ['model.loss_weight', 'iptm', '1.0', '2.5', '鍔犻噸 ipTM 绮惧害鎯╃綒锛屾彁鍗囪璁¤川閲?],
        ['model.loss_weight', 'pae', '0.3', '0.3', '涓嶅彉'],
        ['model.loss_weight', 'train_recycles', '2', '3', '澧炲己鍙嶉宓屽叆姊害淇″彿'],
        ['model.loss_weight', 'mask_aug_prob', '鈥?, '0.4', '鏂板锛?0% 鎵规搴旂敤闅忔満鎺╃爜'],
        ['model.loss_weight', 'mask_aug_ratio', '鈥?, '[0.2, 1.0]', '鏂板锛氭帺鐮?20%-100% 娈嬪熀'],
        ['train.optimizer', 'lr', '1e-4', '5e-5', '闄嶄綆瀛︿範鐜囦互绋冲畾寰皟'],
        ['train.scheduler', 'patience', '30', '40', '寤堕暱鑰愬績鍊硷紝閬垮厤杩囨棭闄嶅涔犵巼'],
        ['train', 'max_iters', '30000', '20000', '鏁版嵁閲忓鍔狅紝棰勬湡鏇村揩鏀舵暃'],
        ['dataset.train', 'db_path', 'v4 (503鏉?', 'merged_v5 (919鏉?', '+82.7% 璁粌鏁版嵁'],
        ['dataset.val', 'db_path', 'v4 (126鏉?', 'merged_v5 (230鏉?', '+82.5% 楠岃瘉鏁版嵁'],
    ]
)

# ============================================================
# 6. 鏁版嵁闆嗗悎骞?
# ============================================================
doc.add_heading('6. 鏁版嵁闆嗗悎骞?, level=1)

doc.add_heading('6.1 鑴氭湰锛歮erge_v5_dataset.py锛堟柊寤猴級', level=2)
doc.add_paragraph(
    '鍔犺浇 V4 鏁版嵁锛?03 璁粌 + 126 楠岃瘉锛夊拰鐤剧梾 v2 鏁版嵁锛?45 鏉★級锛屾寜姘ㄥ熀閰稿簭鍒楀幓閲嶅悗锛岃繘琛?80/20 闅忔満鎵撲贡鍒嗗壊銆?
)

add_table(
    ['鎸囨爣', '鏁板€?],
    [
        ['V4 鏉ユ簮鏉＄洰', '629锛?03璁粌 + 126楠岃瘉锛?],
        ['鐤剧梾 v2 鏉ユ簮鏉＄洰', '545'],
        ['鍘婚噸鍓嶆€绘暟', '1,174'],
        ['鎸夊簭鍒楀幓閲嶇Щ闄?, '25'],
        ['鏈€缁堝敮涓€鏉＄洰', '1,149'],
        ['璁粌闆嗭紙80%锛?, '919'],
        ['楠岃瘉闆嗭紙20%锛?, '230'],
        ['鐤剧梾鏍囩鏉＄洰', '520锛?19璁粌 + 101楠岃瘉锛?],
        ['IDP 鏍囩鏉＄洰', '37'],
        ['鐩告瘮 V4 澧為暱', '+82.7%锛?29 鈫?1,149锛?],
    ]
)

# ============================================================
# 7. V5 璁粌
# ============================================================
doc.add_heading('7. V5 璁粌鎵ц璁板綍', level=1)

doc.add_heading('7.1 闃舵涓€锛氭棫鏁版嵁闆嗭紙鍚堝苟鍓嶏紝629鏉★級', level=2)
doc.add_paragraph(
    '鍚姩鏃堕棿 2026-05-17 02:09銆備娇鐢ㄥ師濮?merged_v5锛?29 鏉★紝鏈寘鍚柧鐥?v2 鎵╁厖鏁版嵁锛夈€?
    '棣栨灏濊瘯鍦?02:03 宕╂簝锛圲nboundLocalError 閿欒锛屽凡淇锛夈€傜浜屾灏濊瘯杩愯绾?5,000 娆¤凯浠ｅ悗涓锛?
    '浠ヤ究浣跨敤鎵╁厖鍚庣殑鐤剧梾鏁版嵁闆嗚繘琛屽悎骞躲€?
)

add_table(
    ['杩唬', '楠岃瘉鎹熷け', 'pLDDT', 'ipTM', 'PAE', '澶囨敞'],
    [
        ['100', '0.0607', '0.0174', '0.0120', '0.0444', '鍒濆鏈€浣?],
        ['400', '0.0535', '0.0147', '0.0100', '0.0457', '鏂版渶浣?],
        ['500', '0.0526', '0.0147', '0.0101', '0.0421', '鏂版渶浣?],
        ['2500锛堢害锛?, '0.0391', '鈥?, '鈥?, '鈥?, '杩涘叆骞冲彴鏈?],
        ['4200', '0.0358', '0.0118', '0.0055', '0.0341', '鏂版渶浣?],
        ['4300', '0.0325', '0.0120', '0.0041', '0.0344', '鏂版渶浣?],
        ['4700锛堟渶缁堬級', '0.0316', '0.0110', '0.0042', '0.0336', '闃舵涓€鏈€浼橈紝鏆傚仠浠ュ悎骞舵暟鎹?],
    ]
)
doc.add_paragraph(
    '闃舵涓€鏈€浣虫鏌ョ偣锛歭ogs/bfn_confidence_combined_v5_2026_05_17__02_09_47/checkpoints/best.pt '
    '锛堥獙璇佹崯澶?0.0316锛岀 4700 娆¤凯浠ｏ級銆傜敤浣滀簡闃舵浜岀殑鏉冮噸鍒濆鍖栥€?
)

doc.add_heading('7.2 闃舵浜岋細鎵╁厖鏁版嵁闆嗭紙鍚堝苟鍚庯紝919璁粌/230楠岃瘉锛?, level=2)
doc.add_paragraph(
    '鍚姩鏃堕棿 2026-05-17 02:39銆傚姞杞介樁娈典竴鏈€浣虫鏌ョ偣鏉冮噸锛屼娇鐢ㄥ叏鏂扮殑浼樺寲鍣ㄥ拰璋冨害鍣ㄣ€?
    '浠庡ご寮€濮嬬儹韬紙200 姝ワ級銆傜洰鍓嶆鍦ㄨ缁冧腑銆?
)

add_table(
    ['杩唬', '楠岃瘉鎹熷け', 'pLDDT', 'ipTM', 'PAE', '澶囨敞'],
    [
        ['100', '0.0607', '0.0174', '0.0120', '0.0444', '鍒濆鏈€浣?],
        ['400', '0.0535', '0.0147', '0.0100', '0.0457', '鏂版渶浣?],
        ['500', '0.0526', '0.0147', '0.0101', '0.0421', '鏂版渶浣?],
        ['600', '0.0512', '0.0149', '0.0092', '0.0446', '鏂版渶浣筹紝ipTM 鏀瑰杽鏄庢樉'],
        ['1000', '0.0473', '鈥?, '鈥?, '鈥?, '鎸佺画涓嬮檷'],
        ['1100', '0.0451', '0.0132', '0.0069', '0.0439', '鈽?褰撳墠鏈€浣?鈽?],
        ['1900', '0.0463', '0.0142', '0.0075', '0.0441', '绋冲畾娉㈠姩'],
        ['2000', '0.0529', '0.0145', '0.0101', '0.0439', '娉㈠姩'],
        ['2100', '0.0520', '0.0143', '0.0099', '0.0433', '鈥?],
        ['2200', '0.0572', '0.0148', '0.0117', '0.0440', '灏忓箙鍥炲崌'],
    ]
)
doc.add_paragraph(
    '瓒嬪娍鍒嗘瀽锛氶獙璇佹崯澶变粠 0.0607 涓嬮檷鑷?0.0451锛堥檷骞?25.7%锛夈€俰pTM 楠岃瘉鎹熷け浠?0.0120 闄嶈嚦 0.0069锛堥檷骞?43%锛夈€?
    '鎺╃爜澧炲己姝ｅ湪璧锋晥鈥斺€旇澶氳缁冩鍑虹幇 seq_loss=0.0000锛堝叏鎺╃爜锛夛紝璁粌鎹熷け鍦?0.002~0.89 涔嬮棿澶у箙娉㈠姩銆?
    '璁粌鐩爣锛?0,000 娆¤凯浠ｃ€傞璁℃€昏€楁椂锛氱害 28 灏忔椂锛堟寜姣忔杩唬绾?5-6 绉掍及绠楋級銆?
)

doc.add_heading('7.3 璁粌鍏抽敭瑙傚療', level=2)
for o in [
    '鎺╃爜澧炲己鏈夋晥妯℃嫙璁捐鏉′欢锛氳缁冩崯澶遍殢闅忔満鎺╃爜姣斾緥澶у箙娉㈠姩锛?.002~0.89锛夛紝楂樻帺鐮佹壒娆′腑 ipTM 鎹熷け椋欏崌鑷?0.31銆傝繖灞炰簬姝ｅ父涓旀湡鏈涚殑鐜拌薄鈥斺€旀ā鍨嬫鍦ㄥ涔犱粠閮ㄥ垎淇℃伅涓娴?ipTM銆?,
    '绾?40% 鐨勮缁冩鏄剧ず seq_loss=0.0000锛屼笌 mask_aug_prob=0.4 鐨勮缃惢鍚堛€傚綋鎵€鏈夋畫鍩鸿鎺╃爜鏃讹紝娌℃湁搴忓垪闇€瑕侀娴嬨€?,
    '楠岃瘉鎹熷け鍦?0.045~0.065 涔嬮棿娉㈠姩灞炰簬姝ｅ父鑼冨洿銆傛墿鍏呯殑 18 涓柧鐥呯被鍒紩鍏ヤ簡缁撴瀯鎬ц川鍚勫紓鐨勮泲鐧斤紝妯″瀷闇€瑕佹洿澶氳凯浠ｆ潵閫傚簲銆?,
    '瀛︿範鐜囧湪鐑韩鍚庣ǔ瀹氬湪 5e-5銆侾lateau 璋冨害鍣紙patience=40锛屽嵆 4,000 娆¤凯浠ｏ級灏嗗湪楠岃瘉鎹熷け鍋滄粸 4,000 姝ュ悗灏嗗涔犵巼鍑忓崐銆?,
    'pair_feat 褰掗浂鏄叧閿璁★細濡傛灉涓嶅綊闆讹紝缂栫爜鍣ㄥ彲閫氳繃閰嶅鐗瑰緛鎺ㄦ柇琚帺鐮佹畫鍩虹殑缁撴瀯锛岄儴鍒嗘姷娑堟帺鐮佸寮虹殑鏁堟灉銆?,
]:
    doc.add_paragraph(o, style='List Bullet')

# ============================================================
# 8. 鐩綍
# ============================================================
doc.add_heading('8. 璁粌鏃ュ織鐩綍涓€瑙?, level=1)
add_table(
    ['鐩綍', '鐢ㄩ€?, '鏈€浣抽獙璇佹崯澶?, '鐘舵€?],
    [
        ['logs/bfn_confidence_combined_v4_2026_05_16__23_04_14', 'V4 鍩虹嚎璁粌', '0.0283锛堢26600姝ワ級', '宸插畬鎴愶紙30K姝ワ級'],
        ['logs/bfn_confidence_combined_v5_2026_05_17__02_03_24', 'V5 闃舵涓€锛堝穿婧冿級', '鈥?, '澶辫触锛圲nboundLocalError锛?],
        ['logs/bfn_confidence_combined_v5_2026_05_17__02_09_47', 'V5 闃舵涓€锛堜慨澶嶅悗锛?, '0.0316锛堢4700姝ワ級', '宸蹭腑姝紙绛夊緟鍚堝苟锛?],
        ['logs/bfn_confidence_combined_v5_2026_05_17__02_39_22', 'V5 闃舵浜岋紙鎵╁厖鏁版嵁锛?, '0.0451锛堢1100姝ワ級', '璁粌涓?..'],
    ]
)

# ============================================================
# 9. 鏂囦欢娓呭崟
# ============================================================
doc.add_heading('9. 淇敼/鍒涘缓鏂囦欢娓呭崟', level=1)
add_table(
    ['鏂囦欢', '鎿嶄綔', '璇存槑'],
    [
        ['disorderflow/modules/bfn/core.py', '淇敼', '鏂板绾?5琛岋細鎺╃爜澧炲己 + pair_feat 褰掗浂閫昏緫'],
        ['build_disease_dataset_v2.py', '鏂板缓', '19涓煡璇€?0-400aa銆乁niProt鍒嗛〉銆佸閲忎繚瀛?],
        ['merge_v5_dataset.py', '鏂板缓', 'v4+鐤剧梾v2鍚堝苟銆佹寜搴忓垪鍘婚噸銆?0/20鍒嗗壊'],
        ['configs/train/bfn_confidence_combined_v5.yml', '鏂板缓', 'V5璁粌閰嶇疆锛歩pTM=2.5銆佹帺鐮佸寮恒€乴r=5e-5銆乺ecycles=3'],
        ['data/confidence_dataset_disease_v2/confidence_disease.lmdb', '鐢熸垚', '545鏉★紝927 MB锛?8涓被鍒?],
        ['data/confidence_merged_v5/confidence_train.lmdb', '鐢熸垚', '919鏉¤缁冩暟鎹?],
        ['data/confidence_merged_v5/confidence_val.lmdb', '鐢熸垚', '230鏉￠獙璇佹暟鎹?],
    ]
)

# ============================================================
# 10. 鍚庣画宸ヤ綔
# ============================================================
doc.add_heading('10. 寰呭畬鎴愬伐浣?, level=1)
for i, item in enumerate([
    '绛夊緟 V5 闃舵浜岃缁冨畬鎴愶紙棰勮绾?28 灏忔椂锛?0K 娆¤凯浠ｏ級',
    '鍦ㄩ樁娈典簩鏈€浣虫鏌ョ偣涓婅繍琛?evaluate_confidence.py锛屾祴閲?pLDDT/ipTM Pearson 鐩稿叧绯绘暟',
    '杩愯 design_8proteins_save.py锛岄獙璇佽璁℃椂 ipTM 鏄惁浠?0.695 鎻愬崌鍒?0.75+',
    '鏇存柊 app_config.yaml锛屽皢妫€鏌ョ偣璺緞鎸囧悜 V5 鏈€浣虫ā鍨?,
    '涓?V5 闃舵浜岀敓鎴?training_log.md锛屽苟灏嗗叾瑙ｆ瀽鐗堟湰涓婁紶鍒?鏃ュ織 鐩綍',
    '鑻ラ樁娈典簩瀹屾垚鍚?ipTM 浠嶄綆浜?0.75锛氳€冭檻鍗囩骇 ipTM 澶撮儴鏋舵瀯锛堟柟妗?B锛?,
], 1):
    doc.add_paragraph(f'{i}. {item}')

# ============================================================
# 淇濆瓨
# ============================================================
doc.save(OUTPUT)
print(f'鎶ュ憡宸蹭繚瀛樿嚦锛歿OUTPUT}')
print(f'鏂囦欢澶у皬锛歿os.path.getsize(OUTPUT) / 1024:.1f} KB')
