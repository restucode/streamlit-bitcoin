import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Optional: joblib for persistent cache (jika mau dipakai)
import joblib

# ------------ MAIN STREAMLIT APP -------------
st.set_page_config(layout="wide", page_title="Bitcoin Sentiment Analysis Streamlit")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'createdAt' in df.columns:
        try:
            df['createdAt'] = pd.to_datetime(df['createdAt'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
        except:
            df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        df['year'] = df['createdAt'].dt.year
    return df

def main():
    st.sidebar.header("Konfigurasi Dataset & Model")
    input_file = st.sidebar.text_input("Path File Data CSV", "./bitcoin2225_pelabelan_embedding.csv")
        
    # HEADER UTAMA di atas tab
    st.markdown("""
        <h2 style="text-align:center; color:#2d3e50;">
            Klasifikasi Unggahan Terkait Bitcoin dan <br>
            Perbandingan Akurasi Algoritma <span style="color:#feb019;">XGBoost</span> dan <span style="color:#7fd8be;">Random Forest</span>
        </h2>
        <hr>
    """, unsafe_allow_html=True)

    # =========== LOAD DATA ===========
    result_df = load_data(input_file)
    emb_cols = [col for col in result_df.columns if col.startswith('emb_')]
    X = result_df[emb_cols].values
    y = result_df['label'].values
    label_names = {0: "Negatif", 1: "Positif"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Distribusi Data", "Sentimen per Tahun", "Kata Populer & Wordcloud",
        "Evaluasi Uji", "Performa Data Latih", "Performa Data Uji", "Margin Error CV"
    ])
    # ----------------------------------------------------------------------
    with tab1:
        st.header("Distribusi Data: Label & Split")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribusi Label:**")
            vc = pd.Series(y).value_counts()
            vc.index = vc.index.map(label_names)
            df_vc = vc.to_frame('jumlah_data')

            # Tambahkan baris total
            total = pd.DataFrame({'jumlah_data': [df_vc['jumlah_data'].sum()]}, index=['Total'])
            df_vc_total = pd.concat([df_vc, total])

            st.dataframe(df_vc_total)
            st.markdown("**Distribusi Split Train/Uji:**")
            st.write(
                f"Train: {len(X_train)},  Test: {len(X_test)},  Rasio: {len(X_train)/len(X):.2f}:{len(X_test)/len(X):.2f}"
            )
            # ----------- Split DataFrame Train & Test untuk Tampilan -----------
        # Simpan index sebelum split supaya hasil train-test sesuai urutan data asli:
        idx_full = np.arange(len(result_df))
        idx_train, idx_test = train_test_split(idx_full, test_size=0.2, stratify=y, random_state=42)
        train_df = result_df.iloc[idx_train].copy()
        test_df  = result_df.iloc[idx_test].copy()

        st.markdown("### Data Train dan Data Test Hasil Split")
        tab_train, tab_test = st.tabs(["Data Train", "Data Test"])

        with tab_train:
            st.subheader("Data Train 📚")
            label_list = ["All"] + [str(l) for l in sorted(train_df['label'].unique())]
            label_filter = st.selectbox("Filter Label (Train)", label_list, key="filter_train_tab")
            if label_filter == "All":
                display_df = train_df
            else:
                display_df = train_df[train_df['label'] == int(label_filter)]
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=350)

        with tab_test:
            st.subheader("Data Test 📝")
            label_list = ["All"] + [str(l) for l in sorted(test_df['label'].unique())]
            label_filter = st.selectbox("Filter Label (Test)", label_list, key="filter_test_tab")
            if label_filter == "All":
                display_df = test_df
            else:
                display_df = test_df[test_df['label'] == int(label_filter)]
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=350)
            with col2:
                fig, axes = plt.subplots(1,2,figsize=(9,3))
                sns.barplot(
                    x=[label_names[i] for i in range(2)],
                    y=pd.Series(y_train).value_counts().sort_index(), ax=axes[0], palette="Set2"
                )
                axes[0].set_title("Train sebelum SMOTE")
                sns.barplot(
                    x=[label_names[i] for i in range(2)],
                    y=pd.Series(y_train_smote).value_counts().sort_index(), ax=axes[1], palette="Set1"
                )
                axes[1].set_title("Train setelah SMOTE")
                for ax in axes:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    for i, bar in enumerate(ax.patches):
                        bar_val = int(bar.get_height())
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), bar_val, ha='center', va='bottom')
                st.pyplot(fig)
    # ----------------------------------------------------------------------
    with tab2:
        st.header("Distribusi Data Sentimen per Tahun")
        if 'year' in result_df.columns:
            sentimen_per_tahun = result_df.groupby(['year', 'label']).size().unstack(fill_value=0)
            sentimen_per_tahun = sentimen_per_tahun.rename(columns={0: 'Negatif', 1: 'Positif'})
            fig, ax = plt.subplots(figsize=(10,5))
            sentimen_per_tahun.plot(kind='bar', ax=ax, color=['#EC7063', '#5DADE2'])
            ax.set_ylabel('Jumlah Data')
            st.pyplot(fig)
            st.dataframe(sentimen_per_tahun)
        else:
            st.warning("Kolom Tahun tidak tersedia atau gagal diparse.")

    # ----------------------------------------------------------------------
    with tab3:
        st.header("10 Kata Tersering & Wordcloud per Sentimen")
        text_col = "teks"
        if text_col in result_df.columns:
            pos_texts = result_df[result_df['label']==1][text_col].astype(str).dropna()
            neg_texts = result_df[result_df['label']==0][text_col].astype(str).dropna()
            # POSITIF
            cv = CountVectorizer(stop_words='english')
            pos_corpus = " ".join(pos_texts.tolist()).lower().translate(str.maketrans('', '', string.punctuation))
            X_pos = cv.fit_transform([pos_corpus])
            pos_word_freq = dict(zip(cv.get_feature_names_out(), X_pos.toarray()[0]))
            pos_common = Counter(pos_word_freq).most_common(10)
            fig1, ax1 = plt.subplots(figsize=(6,3))
            sns.barplot(x=[x[1] for x in pos_common], y=[x[0] for x in pos_common], ax=ax1, palette="crest")
            st.subheader(":green[Positif]")
            st.pyplot(fig1)
            wc = WordCloud(width=400, height=250, background_color="white").generate(pos_corpus)
            st.image(wc.to_array(), use_container_width=True)
            # NEGATIF
            cv = CountVectorizer(stop_words='english')
            neg_corpus = " ".join(neg_texts.tolist()).lower().translate(str.maketrans('', '', string.punctuation))
            X_neg = cv.fit_transform([neg_corpus])
            neg_word_freq = dict(zip(cv.get_feature_names_out(), X_neg.toarray()[0]))
            neg_common = Counter(neg_word_freq).most_common(10)
            fig2, ax2 = plt.subplots(figsize=(6,3))
            sns.barplot(x=[x[1] for x in neg_common], y=[x[0] for x in neg_common], ax=ax2, palette="rocket")
            st.subheader(":red[Negatif]")
            st.pyplot(fig2)
            wc2 = WordCloud(width=400, height=250, background_color="black", colormap="Reds").generate(neg_corpus)
            st.image(wc2.to_array(), use_container_width=True)
        else:
            st.warning("Kolom 'teks' tidak tersedia untuk visualisasi kata.")
    # ----------------------------------------------------------------------
    with tab4:
        st.header("Evaluasi Model pada Data Uji (Test Set)")
        st.info("Menggunakan parameter terbaik: RF (n_estimators=200, max_depth=10, min_samples_split=2), XGB (n_estimators=200, max_depth=3, learning_rate=0.1)")
        @st.cache_resource
        def train_rf():
            rf = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42))
            ])
            return rf.fit(X_train, y_train)
        @st.cache_resource
        def train_xgb():
            xgb = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42))
            ])
            return xgb.fit(X_train, y_train)
        rf_pipeline = train_rf()
        xgb_pipeline = train_xgb()

        # MODIFIKASI DI BAWAH INI
        def eval_model(model, mdlname):
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec_pos  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_pos   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            prec_neg = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            rec_neg  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            f1_neg   = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
            with st.expander(f"Confusion Matrix & Report ({mdlname})", expanded=True):
                st.text(classification_report(
                    y_test, y_pred,
                    target_names=["Negatif","Positif"],
                    zero_division=0))
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(
                    model, X_test, y_test,
                    display_labels=["Negatif","Positif"], ax=ax)
                st.pyplot(fig)
            return acc, prec_pos, rec_pos, f1_pos, prec_neg, rec_neg, f1_neg, y_pred

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Random Forest")
            rf_acc, rf_prec_pos, rf_rec_pos, rf_f1_pos, rf_prec_neg, rf_rec_neg, rf_f1_neg, rf_pred = eval_model(rf_pipeline, "RandomForest")
        with colB:
            st.subheader("XGBoost")
            xgb_acc, xgb_prec_pos, xgb_rec_pos, xgb_f1_pos, xgb_prec_neg, xgb_rec_neg, xgb_f1_neg, xgb_pred = eval_model(xgb_pipeline, "XGBoost")
        st.markdown("---")

        # Tambahkan kolom akurasi, presisi, recall, f1 untuk positif maupun negatif
        report = pd.DataFrame({
            "Akurasi":       [rf_acc, xgb_acc],
            "Presisi (+)":   [rf_prec_pos, xgb_prec_pos],
            "Recall (+)":    [rf_rec_pos, xgb_rec_pos],
            "F1 (+)":        [rf_f1_pos, xgb_f1_pos],
            "Presisi (-)":   [rf_prec_neg, xgb_prec_neg],
            "Recall (-)":    [rf_rec_neg, xgb_rec_neg],
            "F1 (-)":        [rf_f1_neg, xgb_f1_neg],
        }, index=["Random Forest", "XGBoost"])

        st.write("**Metrik Performa di Test Set (Per Sentimen)**")
        st.dataframe(report.round(4))
        
        # Buat DataFrame hasil uji: label, prediksi, teks (atau fitur lain), dan status (TP, TN, FP, FN)
        def get_eval_table(y_true, y_pred, df_test):
            result = df_test.copy().reset_index(drop=True)
            result['True Label'] = y_true
            result['Prediksi'] = y_pred
            condlist = [
                (result['True Label']==1) & (result['Prediksi']==1),
                (result['True Label']==0) & (result['Prediksi']==0),
                (result['True Label']==0) & (result['Prediksi']==1),
                (result['True Label']==1) & (result['Prediksi']==0),
            ]
            choicelist = ['TP', 'TN', 'FP', 'FN']
            result['Pred_Type'] = np.select(condlist, choicelist, default='Other')
            return result

        # DataFrame test sudah didefinisikan sebagai `test_df` (lihat tab1 kode kamu)
        # Filter test_df agar urutannya sama dengan y_test

        def append_total_row(df):
            total_row = {col: '' for col in df.columns}
            total_row['teks'] = 'TOTAL'
            total_row['Pred_Type'] = len(df)
            return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        test_df_reset = test_df.reset_index(drop=True)
        

        st.markdown("### Tabel Hasil Prediksi Data Uji")

        colrf, colxgb = st.columns(2)

        with colrf:
            st.write("#### Random Forest")
            eval_rf = get_eval_table(y_test, rf_pred, test_df_reset)
            rf_pred_types = ['All'] + list(eval_rf['Pred_Type'].unique())
            rf_ptype = st.selectbox("Filter Prediksi RF (TP/TN/FP/FN)", rf_pred_types, key='rf_ptype')
            if rf_ptype == "All":
                rf_show = eval_rf
            else:
                rf_show = eval_rf[eval_rf['Pred_Type'] == rf_ptype]
            st.dataframe(append_total_row(rf_show[['teks','True Label','Prediksi','Pred_Type']]), use_container_width=True)
            csv_rf = rf_show.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil RF (.csv)", csv_rf, "hasil_rf_test.csv")
            

        with colxgb:
            st.write("#### XGBoost")
            eval_xgb = get_eval_table(y_test, xgb_pred, test_df_reset)
            xgb_pred_types = ['All'] + list(eval_xgb['Pred_Type'].unique())
            xgb_ptype = st.selectbox("Filter Prediksi XGB (TP/TN/FP/FN)", xgb_pred_types, key='xgb_ptype')
            if xgb_ptype == "All":
                xgb_show = eval_xgb
            else:
                xgb_show = eval_xgb[eval_xgb['Pred_Type'] == xgb_ptype]
            st.dataframe(append_total_row(xgb_show[['teks','True Label','Prediksi','Pred_Type']]), use_container_width=True)
            csv_xgb = xgb_show.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil XGB (.csv)", csv_xgb, "hasil_xgb_test.csv")
    # ----------------------------------------------------------------------
    with tab5:
        st.header("Performa Model pada Data Latih (Train Set: Before/After SMOTE)")
        @st.cache_resource
        def get_train_scores():
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)
            xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
            # Sebelum SMOTE
            rf.fit(X_train, y_train)
            xgb.fit(X_train, y_train)
            rf_pred_train = rf.predict(X_train)
            xgb_pred_train = xgb.predict(X_train)
            # Sesudah SMOTE
            rf_smote = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)
            xgb_smote = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
            rf_smote.fit(X_train_smote, y_train_smote)
            xgb_smote.fit(X_train_smote, y_train_smote)
            rf_pred_train_smote = rf_smote.predict(X_train_smote)
            xgb_pred_train_smote = xgb_smote.predict(X_train_smote)
            return rf_pred_train, xgb_pred_train, rf_pred_train_smote, xgb_pred_train_smote
        (rf_pred_train, xgb_pred_train, rf_pred_train_smote, xgb_pred_train_smote) = get_train_scores()
        def metric_scores(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            return acc, prec, rec, f1
        rf_train_before_scores = metric_scores(y_train, rf_pred_train)
        xgb_train_before_scores = metric_scores(y_train, xgb_pred_train)
        rf_train_after_scores = metric_scores(y_train_smote, rf_pred_train_smote)
        xgb_train_after_scores = metric_scores(y_train_smote, xgb_pred_train_smote)
        train_report_df = pd.DataFrame({
            'Akurasi':  [rf_train_before_scores[0], rf_train_after_scores[0], xgb_train_before_scores[0], xgb_train_after_scores[0]],
            'Presisi':  [rf_train_before_scores[1], rf_train_after_scores[1], xgb_train_before_scores[1], xgb_train_after_scores[1]],
            'Recall':   [rf_train_before_scores[2], rf_train_after_scores[2], xgb_train_before_scores[2], xgb_train_after_scores[2]],
            'F1 Score': [rf_train_before_scores[3], rf_train_after_scores[3], xgb_train_before_scores[3], xgb_train_after_scores[3]]
        }, index=['Random Forest Sebelum SMOTE', 'Random Forest Sesudah SMOTE',
                  'XGBoost Sebelum SMOTE', 'XGBoost Sesudah SMOTE'])
        st.dataframe(train_report_df.round(4))
        metrics = ['Akurasi', 'Presisi', 'Recall', 'F1 Score']
        fig,axes = plt.subplots(1,4,figsize=(12,4))
        for i, m in enumerate(metrics):
            axes[i].bar(train_report_df.index, train_report_df[m], color=['cornflowerblue','mediumseagreen','tomato','goldenrod'])
            axes[i].set_ylim(0,1)
            axes[i].set_title(m)
            axes[i].tick_params(axis='x', rotation=90)
        st.pyplot(fig)

    with tab6:
        st.header("Performa Model pada Data Uji (Test Set: Before/After SMOTE)")

        # Fungsi metric skor
        def metric_scores(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            return acc, prec, rec, f1

        @st.cache_resource
        def get_test_scores():
            # Model param
            rf_param = dict(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)
            xgb_param = dict(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
            # Sebelum SMOTE
            rf_before = RandomForestClassifier(**rf_param)
            xgb_before = XGBClassifier(**xgb_param)
            rf_before.fit(X_train, y_train)
            xgb_before.fit(X_train, y_train)
            rf_pred_before = rf_before.predict(X_test)
            xgb_pred_before = xgb_before.predict(X_test)
            # Sesudah SMOTE (pakai pipeline)
            rf_pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(**rf_param))
            ]).fit(X_train, y_train)
            xgb_pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', XGBClassifier(**xgb_param))
            ]).fit(X_train, y_train)
            rf_pred_after = rf_pipeline.predict(X_test)
            xgb_pred_after = xgb_pipeline.predict(X_test)
            return rf_pred_before, rf_pred_after, xgb_pred_before, xgb_pred_after

        (rf_pred_before, rf_pred_after, xgb_pred_before, xgb_pred_after) = get_test_scores()
        rf_before_scores  = metric_scores(y_test, rf_pred_before)
        rf_after_scores   = metric_scores(y_test, rf_pred_after)
        xgb_before_scores = metric_scores(y_test, xgb_pred_before)
        xgb_after_scores  = metric_scores(y_test, xgb_pred_after)

        report_df = pd.DataFrame({
            'Akurasi':    [rf_before_scores[0], rf_after_scores[0], xgb_before_scores[0], xgb_after_scores[0]],
            'Presisi':    [rf_before_scores[1], rf_after_scores[1], xgb_before_scores[1], xgb_after_scores[1]],
            'Recall':     [rf_before_scores[2], rf_after_scores[2], xgb_before_scores[2], xgb_after_scores[2]],
            'F1 Score':   [rf_before_scores[3], rf_after_scores[3], xgb_before_scores[3], xgb_after_scores[3]]
        }, index=['Random Forest Sebelum SMOTE', 'Random Forest Sesudah SMOTE',
                'XGBoost Sebelum SMOTE', 'XGBoost Sesudah SMOTE'])

        st.dataframe(report_df.round(4))

        # Visualisasi barplot
        metrics = ['Akurasi', 'Presisi', 'Recall', 'F1 Score']
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        for i, m in enumerate(metrics):
            axes[i].bar(report_df.index, report_df[m], color=['cornflowerblue','mediumseagreen','tomato','goldenrod'])
            axes[i].set_ylim(0, 1)
            axes[i].set_title(m)
            axes[i].tick_params(axis='x', rotation=90)
        st.pyplot(fig)

    # ----------------------------------------------------------------------
    def load_std_result(filename):
        data = np.load(filename)
        return data['train'], data['test']
    with tab7:
        st.header("Margin Error (Std Deviasi via 10-fold CV)")
        st.info("Margin error di-load dari file .npz hasil crossval sebelumnya, sehingga jauh lebih cepat!")

        # Path file hasil crossval/cv margin
        # -- Ubah path jika letak file berbeda pada server/drive kamu
        FOLDER = "./"  # gunakan folder model margin error kamu
        fnames = {
            'rf_std': os.path.join(FOLDER, "rf_std.npz"),
            'rf_smote_std': os.path.join(FOLDER, "rf_smote_std.npz"),
            'xgb_std': os.path.join(FOLDER, "xgb_std.npz"),
            'xgb_smote_std': os.path.join(FOLDER, "xgb_smote_std.npz"),
        }
        names = [
            ("Random Forest\nSebelum SMOTE", 'rf_std'),
            ("Random Forest\nSesudah SMOTE", 'rf_smote_std'),
            ("XGBoost\nSebelum SMOTE", 'xgb_std'),
            ("XGBoost\nSesudah SMOTE", 'xgb_smote_std'),
        ]
        metric_names = ['Akurasi', 'Presisi', 'Recall', 'F1-score']
        result_dict = {}
        for name, k in names:
            if os.path.exists(fnames[k]):
                std_train, std_test = load_std_result(fnames[k])
                result_dict[name] = (std_train, std_test)
            else:
                st.error(f"Hasil margin error model \"{name}\" ({fnames[k]}) tidak ditemukan!")
                result_dict[name] = ([np.nan]*4, [np.nan]*4)

        # Plot
        bar_width = 0.2
        index = np.arange(len(metric_names))
        fig, ax = plt.subplots(1,2,figsize=(16,5))
        for i, (name, (std_train, std_test)) in enumerate(result_dict.items()):
            ax[0].bar(index + i*bar_width, std_train, bar_width, label=name)
            ax[1].bar(index + i*bar_width, std_test, bar_width, label=name)
        ax[0].set_title("Std Deviasi Metrik (Train Fold - CV)")
        ax[1].set_title("Std Deviasi Metrik (Valid/Test Fold - CV)")
        for a in ax:
            a.set_xticks(index + 1.5*bar_width)
            a.set_xticklabels(metric_names, rotation=0)
            a.legend()
            a.set_ylim(0, None)
            a.grid(axis='y', linestyle='--', alpha=0.6)
            a.set_ylabel("Std Deviasi")
        st.pyplot(fig)

        st.write("**Tabel margin error (std) train & valid**")
        data = []
        rows = []
        for name, (std_tr, std_te) in result_dict.items():
            rows.append(f"{name} (Train)")
            data.append(std_tr)
            rows.append(f"{name} (Validasi)")
            data.append(std_te)
        st.dataframe(pd.DataFrame(data, index=rows, columns=metric_names).round(4))

    st.success("Selesai! Silakan eksplor semua tab dan visualisasi interaktif.")

if __name__ == "__main__":
    main()