import os
import urllib.request
import streamlit as st
import pandas as pd
import pulp
import time
from io import BytesIO
from collections import Counter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from translations import translations

st.set_page_config(page_title="Hesaplama Merkezi / Calculation Center", layout="wide")

@st.cache_resource
def setup_fonts():
    font_regular = "Roboto-Regular.ttf"
    font_bold = "Roboto-Bold.ttf"
    url_regular = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
    url_bold = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    try:
        if not os.path.exists(font_regular):
            urllib.request.urlretrieve(url_regular, font_regular)
        if not os.path.exists(font_bold):
            urllib.request.urlretrieve(url_bold, font_bold)
        pdfmetrics.registerFont(TTFont('Roboto', font_regular))
        pdfmetrics.registerFont(TTFont('Roboto-Bold', font_bold))
        return 'Roboto', 'Roboto-Bold'
    except Exception:
        return 'Helvetica', 'Helvetica-Bold'

FONT_REGULAR, FONT_BOLD = setup_fonts()

if "lang" not in st.session_state:
    st.session_state.lang = "🇹🇷 Türkçe"

def t(key, *args):
    # Map the display name with emoji back to the key used in translations dictionary
    lang_map = {
        "🇹🇷 Türkçe": "Türkçe",
        "🇬🇧 English": "English",
        "🇷🇺 Русский": "Русский"
    }
    dict_key = lang_map.get(st.session_state.lang, "Türkçe")
    text = translations[dict_key].get(key, key)
    if args:
        return text.format(*args)
    return text

def solve_cutting_stock_integer(data_list, raw_len, kerf=0):
    L = raw_len + kerf
    item_lengths = [item[0] + kerf for item in data_list]
    original_lengths = [item[0] for item in data_list]
    demands = [item[1] for item in data_list]
    descriptions = [item[2] if len(item) > 2 else "" for item in data_list]
    
    patterns = []
    for i in range(len(item_lengths)):
        row = [0] * len(item_lengths)
        row[i] = 1
        patterns.append(row)

    max_iter = 300
    for _ in range(max_iter):
        master_prob = pulp.LpProblem("Master_LP", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{j}", lowBound=0) for j in range(len(patterns))]
        master_prob += pulp.lpSum(x)
        
        constraints = []
        for i in range(len(item_lengths)):
            c = pulp.lpSum(x[j] * patterns[j][i] for j in range(len(patterns))) >= demands[i]
            master_prob += c
            constraints.append(c)

        master_prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if master_prob.status != 1: break
        
        shadow_prices = [c.pi for c in constraints]

        sub_prob = pulp.LpProblem("Sub_Problem", pulp.LpMaximize)
        a = [pulp.LpVariable(f"a_{i}", lowBound=0, cat='Integer') for i in range(len(item_lengths))]
        sub_prob += pulp.lpSum(a[i] * shadow_prices[i] for i in range(len(item_lengths)))
        sub_prob += pulp.lpSum(a[i] * item_lengths[i] for i in range(len(item_lengths))) <= L
        
        sub_prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if 1 - (pulp.value(sub_prob.objective) or 0) >= -1e-7:
            break

        new_pat = [int(a[i].varValue) for i in range(len(item_lengths))]
        if new_pat in patterns: break
        patterns.append(new_pat)

    final_prob = pulp.LpProblem("Final_Integer_Problem", pulp.LpMinimize)
    x_int = [pulp.LpVariable(f"x_int_{j}", lowBound=0, cat='Integer') for j in range(len(patterns))]
    final_prob += pulp.lpSum(x_int)
    for i in range(len(item_lengths)):
        final_prob += pulp.lpSum(x_int[j] * patterns[j][i] for j in range(len(patterns))) >= demands[i]

    final_prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    total_used = int(pulp.value(final_prob.objective))
    
    details = []
    for j, var in enumerate(x_int):
        if var.varValue > 0:
            count = int(var.varValue)
            pat_desc = []
            pat_len = 0
            for idx, val in enumerate(patterns[j]):
                if val > 0:
                    desc_str = f" ({descriptions[idx]})" if descriptions[idx] else ""
                    pat_desc.append(f"{val}x {original_lengths[idx]}mm{desc_str}")
                    pat_len += val * item_lengths[idx]
            
            details.append({
                "count": count,
                "pattern_str": " + ".join(pat_desc),
                "used_len": pat_len,
                "waste": L - pat_len
            })
            
    return total_used, details

def solve_first_fit_decreasing(data_list, raw_len, kerf=0):
    L = raw_len + kerf
    all_items = []
    for item in data_list:
        l_eff = item[0] + kerf
        l_orig = item[0]
        d = item[1]
        desc = item[2] if len(item) > 2 else ""
        all_items.extend([(l_eff, l_orig, desc)] * d)
    
    all_items.sort(key=lambda x: x[0], reverse=True)
    
    bins = [] 
    
    for item in all_items:
        placed = False
        for b in bins:
            if b['remaining'] >= item[0]:
                b['remaining'] -= item[0]
                b['items'].append(item)
                placed = True
                break
        
        if not placed:
            bins.append({
                'remaining': L - item[0],
                'items': [item]
            })
    
    bin_contents = [tuple(sorted(b['items'], key=lambda x: x[0], reverse=True)) for b in bins]
    bin_counts = Counter(bin_contents)
    
    details = []
    for content, count in bin_counts.items():
        orig_items = [(x[1], x[2]) for x in content]
        item_counts = Counter(orig_items)
        pat_desc = [f"{c}x {l}mm ({desc})" if desc else f"{c}x {l}mm" for ((l, desc), c) in item_counts.items()]
        used_len = sum([x[0] for x in content])
        
        details.append({
            "count": count,
            "pattern_str": " + ".join(pat_desc),
            "used_len": used_len,
            "waste": L - used_len
        })
        
    return len(bins), details

def create_visual_pdf(details, r_len, waste, res1_total, project_title="", parca_listesi=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    y_pos = height - 20*mm

    c.setFont(FONT_BOLD, 20)
    title_text = t("cut_list")
    if project_title:
        title_text += f" - {project_title}"
    c.drawCentredString(width/2, height - 25*mm, title_text)
    
    c.setFont(FONT_REGULAR, 12)
    info_text = t("profile_info_pdf", r_len, res1_total, waste)
    c.drawCentredString(width/2, height - 32*mm, info_text)
    
    c.setStrokeColor(colors.grey)
    c.line(15*mm, height - 40*mm, width - 15*mm, height - 40*mm)
    
    y_pos = height - 48*mm
    
    # if parca_listesi:
    #     summary_title = "Kesim Özeti:" if st.session_state.lang == "🇹🇷 Türkçe" else ("Cutting Summary:" if st.session_state.lang == "🇬🇧 English" else "Сводка резки:")
    #     c.setFont(FONT_BOLD, 12)
    #     c.drawString(15*mm, y_pos, summary_title)
    #     y_pos -= 6*mm
        
    #     c.setFont(FONT_REGULAR, 10)
    #     col1_x = 15*mm
    #     col2_x = 105*mm
    #     current_col = 1
        
    #     for p in parca_listesi:
    #         p_len = p[0]
    #         p_qty = p[1]
    #         p_desc = p[2] if len(p) > 2 else ""
    #         desc_str = f" ({p_desc})" if p_desc else ""
    #         line_text = f"• {p_len}mm{desc_str} : {p_qty} Adet / Pcs"
            
    #         if current_col == 1:
    #             c.drawString(col1_x, y_pos, line_text)
    #             current_col = 2
    #         else:
    #             c.drawString(col2_x, y_pos, line_text)
    #             current_col = 1
    #             y_pos -= 5*mm
                
    #     if current_col == 2:
    #         y_pos -= 5*mm
            
    #     c.setStrokeColor(colors.grey)
    #     c.line(15*mm, y_pos - 2*mm, width - 15*mm, y_pos - 2*mm)
    #     y_pos -= 15*mm
    # else:
    y_pos = height - 65*mm
        
    bar_height = 11*mm
    draw_width = width - 30*mm
    scale_factor = draw_width / r_len
    
    c.setFont("Helvetica", 9)
    
    for item in details:
        small_text_toggle = 0
        count = item['count']
        pattern_str = item['pattern_str']
        waste_val = item['waste']
        
        if y_pos < 25*mm:
            c.showPage()
            y_pos = height - 25*mm
            
        cb_y = y_pos + bar_height + 2*mm
        
        title_text = t("profile_count", count)
        c.setFillColor(colors.black)
        c.setFont(FONT_BOLD, 12)
        c.drawString(15*mm, cb_y + 1*mm, title_text)
        
        text_width = c.stringWidth(title_text, FONT_BOLD, 12)
        start_cb_x = 15*mm + text_width + 2*mm
        cb_size = 4*mm
        gap = 2*mm
        
        c.setStrokeColor(colors.black)
        c.setFillColor(colors.white)
        
        checkbox_rows = 1
        drawn_cb_count = 0
        
        for _ in range(count):
            if start_cb_x + cb_size > width - 15*mm:
                start_cb_x = 15*mm + text_width + 2*mm
                cb_y += cb_size + gap
                checkbox_rows += 1
            
            if checkbox_rows > 2:
                remaining = count - drawn_cb_count
                c.setFillColor(colors.black)
                c.setFont(FONT_BOLD, 10)
                c.drawString(width - 18.5*mm, cb_y - cb_size - gap, t("plus_remaining", remaining))
                break
            
            c.rect(start_cb_x, cb_y, cb_size, cb_size, fill=1, stroke=1)
            drawn_cb_count += 1
            start_cb_x += cb_size + gap
        
        c.setStrokeColor(colors.black)
        c.setFillColor(colors.white)
        c.rect(15*mm, y_pos, draw_width, bar_height, fill=1)
        
        current_x = 15*mm
        
        import re
        if pattern_str:
            parts = pattern_str.split(' + ')
            for p in parts:
                try:
                    m = re.match(r"(\d+)x\s+(\d+)mm(?:\s*\((.*?)\))?", p)
                    if m:
                        p_count = int(m.group(1))
                        p_len = int(m.group(2))
                        p_desc = m.group(3) or ""
                    else:
                        adet_part, boy_part = p.split('x ')
                        p_count = int(adet_part)
                        p_len = int(boy_part.replace('mm', ''))
                        p_desc = ""
                    
                    for _ in range(p_count):
                        part_w = p_len * scale_factor
                        
                        c.setFillColor(colors.lightgrey) 
                        c.setStrokeColor(colors.black)
                        c.rect(current_x, y_pos, part_w, bar_height, fill=1)
                        
                        if part_w > 12*mm:
                            c.setFont(FONT_BOLD, 14)
                        elif part_w > 7*mm:
                            c.setFont(FONT_BOLD, 10)
                        else:
                            c.setFont(FONT_BOLD, 9)

                        c.setFillColor(colors.black)
                        text_x = current_x + (part_w / 2)

                        if part_w > 5*mm:
                            text_y = y_pos + (bar_height / 2) - 1.5*mm 
                            if p_desc and part_w > 12*mm:
                                c.drawCentredString(text_x, text_y + 1.5*mm, str(p_len))
                                c.setFont(FONT_REGULAR, 7)
                                
                                max_chars = int(part_w / (1.6 * mm))
                                display_desc = p_desc
                                if len(p_desc) > max_chars:
                                    display_desc = p_desc[:max_chars] + ".."
                                    
                                c.drawCentredString(text_x, text_y - 2.5*mm, display_desc)
                            else:
                                c.drawCentredString(text_x, text_y, str(p_len))
                        else:
                            if small_text_toggle%2 == 0:
                                text_y = y_pos + (bar_height / 2) - 9*mm
                            else:
                                text_y = y_pos + (bar_height / 2) + 6.1*mm
                            small_text_toggle += 1
                            c.drawCentredString(text_x, text_y, str(p_len))
                        
                        current_x += part_w
                except Exception as e:
                    pass
                    
        if waste_val > 0:
            waste_w = waste_val * scale_factor
            
            if waste_w > 1:
                c.setFillColor(colors.white)
                
                c.setFillColor(colors.whitesmoke)
                c.setStrokeColor(colors.black)
                c.setDash(2, 2)
                c.rect(current_x, y_pos, waste_w, bar_height, fill=1)
                c.setDash([])
                
                c.setFillColor(colors.black)
                c.setFont(FONT_REGULAR, 8)
                c.drawString(current_x + 1.5*mm, y_pos + bar_height + 2*mm, t("waste_pdf", waste_val))

        y_pos -= 25*mm
    
    c.save()
    buffer.seek(0)
    return buffer

col_title, col_lang = st.columns([5, 1])

with col_title:
    st.title(t("main_title"))
with col_lang:
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.selectbox("Dil", options=["🇹🇷 Türkçe", "🇬🇧 English", "🇷🇺 Русский"], key="lang", label_visibility="collapsed")

main_col1, main_col2 = st.columns(2)

@st.dialog(t("are_you_sure"))
def clear_table_dialog():
    st.write(t("clear_warning"))
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button(t("yes_clear"), type="primary", width='stretch'):
            st.session_state.df = pd.DataFrame([{"Uzunluk": 0, "Adet": 0, "Açıklama": ""}])
            st.session_state.run_calculation = False
            st.rerun()
    with col_d2:
        if st.button(t("cancel"), width='stretch'):
            st.rerun()

def reset_calculation():
    st.session_state.run_calculation = False

with main_col1:
    st.subheader(t("parts_to_cut"))

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame([])

    with st.expander(t("file_ops")):
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.info(t("export"))
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t("download_csv"),
                data=csv,
                file_name='kesim_listesi.csv',
                mime='text/csv',
                key='download-csv'
            )
            
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.df.to_excel(writer, index=False, sheet_name='Kesim Listesi')
            
            st.download_button(
                label=t("download_excel"),
                data=buffer.getvalue(),
                file_name='kesim_listesi.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download-excel'
            )

        with d_col2:
            st.info(t("import"))
            uploaded_file = st.file_uploader(t("upload_file"), type=['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_new = pd.read_csv(uploaded_file)
                    else:
                        df_new = pd.read_excel(uploaded_file)
                    
                    required_cols = ['Uzunluk', 'Adet']
                    if all(col in df_new.columns for col in required_cols):
                        if st.button(t("load_data")):
                            if 'Açıklama' not in df_new.columns:
                                df_new['Açıklama'] = ""
                            df_new['Açıklama'] = df_new['Açıklama'].fillna("").astype(str)
                            st.session_state.df = df_new[['Uzunluk', 'Adet', 'Açıklama']]
                            st.session_state.run_calculation = False
                            st.success(t("data_loaded"))
                            st.rerun()
                    else:
                        st.error(f"{t('missing_cols')} {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"{t('error')} {e}")

    col_desc_label = "Açıklama" if st.session_state.lang == "🇹🇷 Türkçe" else "Description" if st.session_state.lang == "🇬🇧 English" else "Описание"

    def add_item():
        new_len = st.session_state.get("add_len", 0)
        new_qty = st.session_state.get("add_qty", 0)
        new_desc = st.session_state.get("add_desc", "")
        
        if new_len > 0 and new_qty > 0:
            new_row = pd.DataFrame([{"Uzunluk": new_len, "Adet": new_qty, "Açıklama": new_desc}])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            st.session_state.run_calculation = False
        
    col_add1, col_add2, col_add4, col_add3 = st.columns([2, 2, 3, 1])
    with col_add1:
        st.number_input(t("part_len"), min_value=10, key="add_len")
    with col_add2:
        st.number_input(t("qty"), min_value=1, value=1, key="add_qty")
    with col_add4:
        st.text_input(col_desc_label, key="add_desc")
    with col_add3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button(t("add"), on_click=add_item, width='stretch')

    column_config = {
        "Uzunluk": st.column_config.NumberColumn(
            label=t("len_col"),
            min_value=10, 
            format="%d",
            width="medium"
        ),
        "Adet": st.column_config.NumberColumn(
            label=t("qty_col"),
            min_value=1, 
            format="%d",
            width="medium"
        ),
        "Açıklama": st.column_config.TextColumn(
            label=col_desc_label,
            width="large"
        ),
    }

    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        column_config=column_config,
        width='stretch',
        height=400,
        hide_index=False,
        key="editor"
    )
    
    if not edited_df.equals(st.session_state.df):
        st.session_state.df = edited_df
        st.session_state.run_calculation = False

    total_pieces = 0
    total_types = 0
    if not edited_df.empty:
        total_pieces = edited_df["Adet"].sum()
        total_types = len(edited_df)
        st.info(t("total_info", total_pieces, total_types))

    if total_pieces > 0:
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
        with col_btn1:
            if st.button(t("quick_calc"), type="primary", width='stretch'):
                st.session_state.run_calculation = True
                st.session_state.run_advanced = False
        with col_btn2:
            if st.button(t("adv_calc"), type="secondary", width='stretch'):
                st.session_state.run_calculation = True
                st.session_state.run_advanced = True
    else:
        st.warning(t("empty_warning"))
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
        with col_btn1:
           st.button(t("quick_calc_short"), type="primary", width='stretch', disabled=True)
        with col_btn2:
           st.button(t("adv_calc"), type="secondary", width='stretch', disabled=True)
    
    with col_btn3:
        if st.button(t("clear_table"), type="secondary", width='stretch'):
            clear_table_dialog()

with main_col2:
    st.subheader(t("profile_details"))
    col_proj_label = "Kesim / Profil Başlığı" if st.session_state.lang == "🇹🇷 Türkçe" else ("Cut / Profile Title" if st.session_state.lang == "🇬🇧 English" else "Название профиля")
    project_title = st.text_input(col_proj_label, key="project_title", on_change=reset_calculation)
    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

    input_col1, input_col3 = st.columns(2)
    with input_col1:
        raw_length = st.number_input(t("profile_len"), min_value=10, value=6000, on_change=reset_calculation)
    with input_col3:
        waste_limit = st.number_input(t("waste_limit"), min_value=0, value=0, on_change=reset_calculation)
    
    st.divider()

    st.subheader(t("results"))
    if st.session_state.get("run_calculation"):
        start_time = time.time()
        
        parca_listesi = []
        for index, row in edited_df.iterrows():
            if row['Uzunluk'] and row['Adet']:
                uzunluk = int(row['Uzunluk'])
                adet = int(row['Adet'])
                
                aciklama = str(row.get('Açıklama', '')).strip()
                if aciklama == 'nan': aciklama = ""
                # prevent pattern split break
                aciklama = aciklama.replace('+', '-').replace('x ', '* ')

                parca_listesi.append([uzunluk, adet, aciklama])
        
        parca_listesi.sort(key=lambda x: x[0], reverse=True)

        if parca_listesi == []:
            st.error(t("empty_list"))
            st.stop()

        eff_len = raw_length
        total_needed_len = sum(p[0] * p[1] for p in parca_listesi)
        
        t2 = time.time()
        res2_total, res2_details = solve_first_fit_decreasing(parca_listesi, eff_len, kerf=waste_limit)
        d2 = time.time() - t2
        
        used2 = res2_total * eff_len
        waste2 = used2 - total_needed_len
        waste2_p = (waste2 / used2) * 100 if used2 > 0 else 0
        
        run_advanced = st.session_state.get("run_advanced", False)
        
        if run_advanced:
            t1 = time.time()
            res1_total, res1_details = solve_cutting_stock_integer(parca_listesi, eff_len, kerf=waste_limit)
            d1 = time.time() - t1
            
            used1 = res1_total * eff_len
            waste1 = used1 - total_needed_len
            waste1_p = (waste1 / used1) * 100 if used1 > 0 else 0
            
            col_cmp1, col_cmp2 = st.columns(2)
            
            with col_cmp1:
                st.info(t("method_1"))
                st.metric(t("req_profile"), f"{res1_total} Adet", delta=None)
                st.metric(t("waste_rate"), f"%{waste1_p:.2f}", delta_color="inverse")
                st.caption(t("calc_time", d1))
                
                with st.expander(t("details_m1"), expanded=True):
                    df_res1 = pd.DataFrame(res1_details)
                    if not df_res1.empty:
                        df_res1['Fire (mm)'] = df_res1['waste']
                        st.dataframe(
                            df_res1[['count', 'pattern_str', 'Fire (mm)']].rename(
                                columns={'count': t("qty"), 'pattern_str': t("cut_template"), 'Fire (mm)': t("waste_mm")}
                            ),
                            width='stretch',
                            hide_index=True
                        )

            with col_cmp2:
                st.warning(t("method_2"))
                st.metric(t("req_profile"), f"{res2_total} Adet", delta=f"{res2_total - res1_total} " + t("diff") if res2_total != res1_total else t("equal"), delta_color="inverse")
                st.metric(t("waste_rate"), f"%{waste2_p:.2f}", delta=f"{waste2_p - waste1_p:.2f}% " + t("diff") if waste2_p != waste1_p else None, delta_color="inverse")
                st.caption(t("calc_time", d2))

                with st.expander(t("details_m2")):
                    df_res2 = pd.DataFrame(res2_details)
                    if not df_res2.empty:
                        df_res2['Fire (mm)'] = df_res2['waste']
                        st.dataframe(
                            df_res2[['count', 'pattern_str', 'Fire (mm)']].rename(
                                columns={'count': t("qty"), 'pattern_str': t("cut_template"), 'Fire (mm)': t("waste_mm")}
                            ),
                            width='stretch',
                            hide_index=True
                        )
        else:
            st.warning(t("method_quick"))
            st.metric(t("req_profile"), f"{res2_total} Adet", delta=None)
            st.metric(t("waste_rate"), f"%{waste2_p:.2f}", delta=None, delta_color="inverse")
            st.caption(t("calc_time", d2))

            with st.expander(t("details_quick"), expanded=True):
                df_res2 = pd.DataFrame(res2_details)
                if not df_res2.empty:
                    df_res2['Fire (mm)'] = df_res2['waste']
                    st.dataframe(
                        df_res2[['count', 'pattern_str', 'Fire (mm)']].rename(
                            columns={'count': t("qty"), 'pattern_str': t("cut_template"), 'Fire (mm)': t("waste_mm")}
                        ),
                        width='stretch',
                        hide_index=True
                    )

        end_time = time.time()
        duration = end_time - start_time
        
        st.success(t("calc_done", duration))
        
        st.info(t("used_prof_info", raw_length, waste_limit))

        project_title_val = st.session_state.get("project_title", "")
        
        import re
        safe_title = re.sub(r'[\\/*?:"<>|]', "", project_title_val).strip()
        # Boşlukları alt tireye çevirmek temiz durur
        safe_title = safe_title.replace(" ", "_")
        f_prefix = f"{safe_title}_" if safe_title else "kesim_plani_"

        if run_advanced:
            p_col1, p_col2 = st.columns(2)
            
            with p_col1:
                pdf_buffer_1 = create_visual_pdf(res1_details, raw_length, waste_limit, res1_total, project_title_val, parca_listesi)
                st.download_button(
                    label=t("pdf_m1"),
                    data=pdf_buffer_1,
                    file_name=f"{f_prefix}1.pdf",
                    mime="application/pdf",
                    type="primary",
                    width='stretch'
                )
            
            with p_col2:
                pdf_buffer_2 = create_visual_pdf(res2_details, raw_length, waste_limit, res2_total, project_title_val, parca_listesi)
                st.download_button(
                    label=t("pdf_m2"),
                    data=pdf_buffer_2,
                    file_name=f"{f_prefix}2.pdf",
                    mime="application/pdf",
                    type="secondary",
                    width='stretch'
                )
        else:
            pdf_buffer_2 = create_visual_pdf(res2_details, raw_length, waste_limit, res2_total, project_title_val, parca_listesi)
            st.download_button(
                label=t("pdf_quick"),
                data=pdf_buffer_2,
                file_name=f"{f_prefix}hizli.pdf",
                mime="application/pdf",
                type="primary",
                width='stretch'
            )
