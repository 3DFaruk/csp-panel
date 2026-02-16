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

st.set_page_config(page_title="Hesaplama Merkezi", layout="wide")

def solve_cutting_stock_integer(data_list, raw_len):
    L = raw_len
    item_lengths = [item[0] for item in data_list]
    demands = [item[1] for item in data_list]
    
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
                    pat_desc.append(f"{val}x {item_lengths[idx]}mm")
                    pat_len += val * item_lengths[idx]
            
            details.append({
                "count": count,
                "pattern_str": " + ".join(pat_desc),
                "used_len": pat_len,
                "waste": L - pat_len
            })
            
    return total_used, details

def solve_first_fit_decreasing(data_list, raw_len):
    all_items = []
    for l, d in data_list:
        all_items.extend([l] * d)
    
    all_items.sort(reverse=True)
    
    bins = [] 
    
    for item in all_items:
        placed = False
        for b in bins:
            if b['remaining'] >= item:
                b['remaining'] -= item
                b['items'].append(item)
                placed = True
                break
        
        if not placed:
            bins.append({
                'remaining': raw_len - item,
                'items': [item]
            })
    
    bin_contents = [tuple(sorted(b['items'], reverse=True)) for b in bins]
    bin_counts = Counter(bin_contents)
    
    details = []
    for content, count in bin_counts.items():
        item_counts = Counter(content)
        pat_desc = [f"{c}x {l}mm" for l, c in item_counts.items()]
        used_len = sum(content)
        
        details.append({
            "count": count,
            "pattern_str": " + ".join(pat_desc),
            "used_len": used_len,
            "waste": raw_len - used_len
        })
        
    return len(bins), details

def create_visual_pdf(details, r_len, r_qty, waste, res1_total):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    y_pos = height - 20*mm

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height - 25*mm, "KESIM LISTESI")
    
    c.setFont("Helvetica", 12)
    info_text = f"Profil Boyu: {r_len}mm | Stok: {r_qty} adet | Gereken Profil: {res1_total} adet | Fire Payi: {waste}mm"
    c.drawCentredString(width/2, height - 32*mm, info_text)
    
    c.setStrokeColor(colors.grey)
    c.line(15*mm, height - 40*mm, width - 15*mm, height - 40*mm)
    
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
        
        title_text = f"{count} Adet Profil:"
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(15*mm, cb_y + 1*mm, title_text)
        
        text_width = c.stringWidth(title_text, "Helvetica-Bold", 12)
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
                c.setFont("Helvetica-Bold", 10)
                c.drawString(width - 18.5*mm, cb_y - cb_size - gap, f"+{remaining} ADET")
                break
            
            c.rect(start_cb_x, cb_y, cb_size, cb_size, fill=1, stroke=1)
            drawn_cb_count += 1
            start_cb_x += cb_size + gap
        
        c.setStrokeColor(colors.black)
        c.setFillColor(colors.white)
        c.rect(15*mm, y_pos, draw_width, bar_height, fill=1)
        
        current_x = 15*mm
        
        if pattern_str:
            parts = pattern_str.split(' + ')
            for p in parts:
                try:
                    adet_part, boy_part = p.split('x ')
                    p_count = int(adet_part)
                    p_len = int(boy_part.replace('mm', ''))
                    
                    for _ in range(p_count):
                        part_w = p_len * scale_factor
                        
                        c.setFillColor(colors.lightgrey) 
                        c.setStrokeColor(colors.black)
                        c.rect(current_x, y_pos, part_w, bar_height, fill=1)
                        
                        if part_w > 12*mm:
                            c.setFont("Helvetica-Bold", 14)
                        elif part_w > 7*mm:
                            c.setFont("Helvetica-Bold", 10)
                        else:
                            c.setFont("Helvetica-Bold", 9)

                        c.setFillColor(colors.black)
                        text_x = current_x + (part_w / 2)

                        if part_w > 5*mm:
                            text_y = y_pos + (bar_height / 2) - 1.5*mm 
                        else:
                            if small_text_toggle%2 == 0:
                                text_y = y_pos + (bar_height / 2) - 9*mm
                            else:
                                text_y = y_pos + (bar_height / 2) + 6.1*mm
                            small_text_toggle += 1
                        
                        c.drawCentredString(text_x, text_y, str(p_len))
                        current_x += part_w
                except:
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
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(current_x + 1.5*mm, y_pos + bar_height + 2*mm, f"Fire: {waste_val}")

        y_pos -= 25*mm
    
    c.save()
    buffer.seek(0)
    return buffer

st.title("📊 Hesaplama ve Optimizasyon Paneli")

main_col1, main_col2 = st.columns(2)

@st.dialog("Emin misiniz?")
def clear_table_dialog():
    st.write("Tüm liste silinecek. Bu işlem geri alınamaz.")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("Evet, Sil", type="primary", width='stretch'):
            st.session_state.df = pd.DataFrame([{"Uzunluk": 0, "Adet": 0}])
            st.rerun()
    with col_d2:
        if st.button("İptal", width='stretch'):
            st.rerun()

with main_col1:
    st.subheader("Kesilecek Parçalar")

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame([])

    with st.expander("Dosya İşlemleri (İçe/Dışa Aktar)"):
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.info("💡 **Dışa Aktar**")
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSV Olarak İndir",
                data=csv,
                file_name='kesim_listesi.csv',
                mime='text/csv',
                key='download-csv'
            )
            
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.df.to_excel(writer, index=False, sheet_name='Kesim Listesi')
            
            st.download_button(
                label="Excel Olarak İndir",
                data=buffer.getvalue(),
                file_name='kesim_listesi.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download-excel'
            )

        with d_col2:
            st.info("📥 **İçe Aktar**")
            uploaded_file = st.file_uploader("Dosya Yükle (CSV/Excel)", type=['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_new = pd.read_csv(uploaded_file)
                    else:
                        df_new = pd.read_excel(uploaded_file)
                    
                    required_cols = ['Uzunluk', 'Adet']
                    if all(col in df_new.columns for col in required_cols):
                        if st.button("Verileri Tabloya Yükle"):
                            st.session_state.df = df_new[required_cols]
                            st.success("Veriler başarıyla yüklendi!")
                            st.rerun()
                    else:
                        st.error(f"Dosyada şu sütunlar olmalı: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"Hata: {e}")

    def add_item():
        new_len = st.session_state.get("add_len", 0)
        new_qty = st.session_state.get("add_qty", 0)
        
        if new_len > 0 and new_qty > 0:
            new_row = pd.DataFrame([{"Uzunluk": new_len, "Adet": new_qty}])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        
    col_add1, col_add2, col_add3 = st.columns([2, 2, 1])
    with col_add1:
        st.number_input("Parça Uzunluğu (mm)", min_value=10, key="add_len")
    with col_add2:
        st.number_input("Adet", min_value=1, value=1, key="add_qty")
    with col_add3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button("Ekle", on_click=add_item, width='stretch')

    column_config = {
        "Uzunluk": st.column_config.NumberColumn(
            label="Uzunluk (mm)",
            min_value=10, 
            format="%d",
            width="medium"
        ),
        "Adet": st.column_config.NumberColumn(
            label="Adet",
            min_value=1, 
            format="%d",
            width="medium"
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

    total_pieces = 0
    total_types = 0
    if not edited_df.empty:
        total_pieces = edited_df["Adet"].sum()
        total_types = len(edited_df)
        st.info(f"Toplam Parça Sayısı: **{total_pieces}** | Farklı Parça Türü: **{total_types}**")

    if total_pieces > 0:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Hesaplamayı Başlat", type="primary", width='stretch'):
                st.session_state.run_calculation = True
    else:
        st.warning("⚠️ Lütfen listeye en az bir parça ekleyiniz.")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
           st.button("Hesaplamayı Başlat", type="primary", width='stretch', disabled=True)
    
    with col_btn2:
        if st.button("Tabloyu Temizle", type="secondary", width='stretch'):
            clear_table_dialog()

with main_col2:
    st.subheader("Profil Bilgileri")
    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        raw_length = st.number_input("Profil Uzunluğu (mm)", min_value=10, value=6000)
    with input_col2:
        raw_qty = st.number_input("Profil Adedi", min_value=1, value=100)
    with input_col3:
        waste_limit = st.number_input("Kesim Fire Payı (mm)", min_value=0, value=0)
    
    st.divider()

    st.subheader("Sonuçlar")
    if st.session_state.get("run_calculation"):
        start_time = time.time()
        
        parca_listesi = []
        for index, row in edited_df.iterrows():
            if row['Uzunluk'] and row['Adet']:
                uzunluk = int(row['Uzunluk'])
                adet = int(row['Adet'])

                parca_listesi.append([uzunluk,adet])
        
        parca_listesi.sort(key=lambda x: x[0], reverse=True)

        if parca_listesi == []:
            st.error("Lütfen parça listesi oluşturun.")
            st.stop()

        eff_len = raw_length - waste_limit
        
        t1 = time.time()
        res1_total, res1_details = solve_cutting_stock_integer(parca_listesi, eff_len)
        d1 = time.time() - t1
        
        t2 = time.time()
        res2_total, res2_details = solve_first_fit_decreasing(parca_listesi, eff_len)
        d2 = time.time() - t2
        
        total_needed_len = sum(p[0] * p[1] for p in parca_listesi)
        
        used1 = res1_total * eff_len
        waste1 = used1 - total_needed_len
        waste1_p = (waste1 / used1) * 100 if used1 > 0 else 0
        
        used2 = res2_total * eff_len
        waste2 = used2 - total_needed_len
        waste2_p = (waste2 / used2) * 100 if used2 > 0 else 0
        
        col_cmp1, col_cmp2 = st.columns(2)
        
        with col_cmp1:
            st.info("###### 1. Yöntem: Gelişmiş Optimizasyon (Pulp)")
            st.metric("Gereken Profil", f"{res1_total} Adet", delta=None)
            st.metric("Fire Oranı", f"%{waste1_p:.2f}", delta_color="inverse")
            st.caption(f"Hesaplama Süresi: {d1:.4f} sn")
            
            with st.expander("Detaylar (Yöntem 1)", expanded=True):
                df_res1 = pd.DataFrame(res1_details)
                if not df_res1.empty:
                    df_res1['Fire (mm)'] = df_res1['waste']
                    st.dataframe(
                        df_res1[['count', 'pattern_str', 'Fire (mm)']].rename(
                            columns={'count': 'Adet', 'pattern_str': 'Kesim Şablonu'}
                        ),
                        width='stretch',
                        hide_index=True
                    )

        with col_cmp2:
            st.warning("###### 2. Yöntem: Hızlı Yerleştirme (First Fit)")
            st.metric("Gereken Profil", f"{res2_total} Adet", delta=f"{res2_total - res1_total} Fark" if res2_total != res1_total else "Eşit", delta_color="inverse")
            st.metric("Fire Oranı", f"%{waste2_p:.2f}", delta=f"{waste2_p - waste1_p:.2f}% Fark" if waste2_p != waste1_p else None, delta_color="inverse")
            st.caption(f"Hesaplama Süresi: {d2:.4f} sn")

            with st.expander("Detaylar (Yöntem 2)"):
                df_res2 = pd.DataFrame(res2_details)
                if not df_res2.empty:
                    df_res2['Fire (mm)'] = df_res2['waste']
                    st.dataframe(
                        df_res2[['count', 'pattern_str', 'Fire (mm)']].rename(
                            columns={'count': 'Adet', 'pattern_str': 'Kesim Şablonu'}
                        ),
                        width='stretch',
                        hide_index=True
                    )

        end_time = time.time()
        duration = end_time - start_time
        
        st.success(f"Hesaplama tamamlandı! (Süre: {duration:.4f} saniye)")
        
        st.info(f"Kullanılan Profil: {raw_length} mm | Stok: {raw_qty} adet | Fire Payı: {waste_limit} mm")

        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            pdf_buffer_1 = create_visual_pdf(res1_details, raw_length, raw_qty, waste_limit, res1_total)
            st.download_button(
                label="📄 Yöntem 1 (Optimal) PDF İndir",
                data=pdf_buffer_1,
                file_name="kesim_plani_yontem1.pdf",
                mime="application/pdf",
                type="primary",
                width='stretch'
            )
        
        with p_col2:
            pdf_buffer_2 = create_visual_pdf(res2_details, raw_length, raw_qty, waste_limit, res2_total)
            st.download_button(
                label="📄 Yöntem 2 (Hızlı) PDF İndir",
                data=pdf_buffer_2,
                file_name="kesim_plani_yontem2.pdf",
                mime="application/pdf",
                type="secondary",
                width='stretch'
            )
