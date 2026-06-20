import sys
import os
import re
import importlib.util

import csv
import pandas as pd

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
    QTabWidget, QGroupBox, QFileDialog, QComboBox, QCheckBox
)

import torch

# https://x.com/rasjawa
# https://x.com/DinaDwiiAnggra
# https://x.com/ljnnno

import html

print("CURRENT DIR =", os.getcwd())
print("FILE DIR =", os.path.dirname(os.path.abspath(__file__)))
print("MODEL EXISTS =", os.path.exists("./indobert_smote70"))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "IndoBERT": {
        "model_path": os.path.join(BASE_DIR, "indobert_smote70"),
        "max_length": 128,
    }
}

if importlib.util.find_spec("tweepy") is None:
    tweepy = None
else:
    import tweepy


class ScrapeWorker(QThread):
    # Worker thread untuk men-scrape tweet dari profil X/Twitter
    # - Mengeluarkan sinyal `progress` untuk update teks status
    # - Mengeluarkan sinyal `finished` dengan payload {'comments': [...], 'profile_info': {...}}
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, profile_url, bearer_token=None):
        super().__init__()
        # URL profil yang akan di-scrape
        self.profile_url = profile_url
        # Batasi ke 10 tweet terbaru (UI menampilkan recent 10)
        self.limit = 10  # Always scrape the 10 newest tweets
        # Bearer token default (dapat dioverride saat inisialisasi)
        self.bearer_token = bearer_token or "AAAAAAAAAAAAAAAAAAAAAGHY8AEAAAAAK%2BcBg8LpY%2BUg5rTiWTOfMybAUhg%3D7Xaf5y3dW7OxTlJlH8OkpPeVucJHbZb9myzzcCud63l0uMTq58"

    def run(self):
        from urllib.parse import urlparse

        parsed_url = urlparse(self.profile_url)
        if not parsed_url.path:
            username = None
        else:
            username = parsed_url.path.strip('/').split('/')[-1] or None

        if not username:
            self.progress.emit("❌ Invalid Twitter/X profile URL")
            self.finished.emit({'comments': [], 'profile_info': None})
            return

        self.progress.emit(f"✓ Username: @{username}")
        self.progress.emit("Fetching tweets with Twitter API v2...")

        client = tweepy.Client(bearer_token=self.bearer_token)

        self.progress.emit("Getting user info...")
        user = client.get_user(username=username, user_fields=['name', 'username', 'profile_image_url'])
        if not user.data:
            self.progress.emit(f"❌ User @{username} not found")
            self.finished.emit({'comments': [], 'profile_info': None})
            return

        user_id = user.data.id
        profile_info = {
            'name': getattr(user.data, 'name', username),
            'username': getattr(user.data, 'username', username),
            'avatar_url': getattr(user.data, 'profile_image_url', None),
        }
        self.progress.emit(f"✓ User ID: {user_id}")

        self.progress.emit(f"Fetching latest {self.limit} tweets...")
        tweets = []
        pagination_token = None

        while len(tweets) < self.limit:
            tweets_response = client.get_users_tweets(
                id=user_id,
                max_results=100,
                tweet_fields=['id', 'text', 'created_at'],
                exclude=['retweets', 'replies'],
                pagination_token=pagination_token,
            )

            if not tweets_response.data:
                break

            for tweet in tweets_response.data:
                tweets.append(tweet.text)
                self.progress.emit(f"Scraped {len(tweets)}/{self.limit} tweets")
                if len(tweets) >= self.limit:
                    break

            pagination_token = getattr(tweets_response.meta, 'next_token', None)
            if not pagination_token:
                break

        if not tweets:
            self.progress.emit("❌ No tweets found (might be protected)")
            self.finished.emit({'comments': [], 'profile_info': profile_info})
            return

        if tweets:
            self.progress.emit(f"✓ Successfully scraped {len(tweets)} tweets!")
        else:
            self.progress.emit("No tweets found")

        self.finished.emit({'comments': tweets, 'profile_info': profile_info})


class AnalysisWorker(QThread):
    # Worker untuk melakukan inference model pada daftar komentar.
    # Keluaran: sinyal `progress` (string) selama proses, dan `finished` dengan list hasil prediksi.
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, comments, model_config):
        super().__init__()
        self.comments = comments
        self.model_config = model_config  # Dict from MODEL_CONFIGS

    def run(self):
        # Hindari import tensorflow yang bisa bentrok
        sys.modules['tensorflow'] = None
        sys.modules['tensorflow.python'] = None
        sys.modules['tensorflow.python.pywrap_tensorflow'] = None

        self.progress.emit('⏳ Importing')
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.progress.emit('✓ Imports successful')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = self.model_config['model_path']
        max_length = self.model_config['max_length']
        

        self.progress.emit('🔧 Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.progress.emit('✓ Tokenizer loaded from base model')

        self.progress.emit('🔧 Loading IndoBERT model...')

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        model.to(device)
        model.eval()

        self.progress.emit(f'✓ Model ready on {device}')
        self.progress.emit(f'📝 Starting inference on {len(self.comments)} comments...')

        results = []
        self.progress.emit(f'🔄 Processing 1-{len(self.comments)} / {len(self.comments)}')

        batch_texts = [preprocess_text(t) for t in self.comments]

        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.view(-1)

        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).long()

        probs_cpu = probs.cpu().numpy()
        preds_cpu = preds.cpu().numpy()

        for j, text in enumerate(self.comments):
            p1 = float(probs_cpu[j])
            p0 = 1.0 - p1

            pred_label = int(preds_cpu[j])

            results.append({
                'comment': text,
                'preprocessed_text': batch_texts[j],
                'pred_label': pred_label,
                'prob_0': p0,
                'prob_1': p1,
            })

        self.progress.emit(f'✓ Analysis complete: {len(results)} results')
        self.finished.emit(results)

class TwitterScraperWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('X/Tweeter suicide detection')
        self.setGeometry(200, 200, 900, 600)
        self.setup_ui()
        self.worker = None
        self.profile_info = None
        self.df = None
        self.prediction_results = []

    def predict_manual_text(self):
        text = self.manual_text.toPlainText().strip()

        if not text:
            QMessageBox.warning(
                self,
                "Peringatan",
                "Masukkan teks terlebih dahulu!"
            )
            return

        self.manual_output.clear()
        self.tab_widget.setCurrentIndex(0)

        model_config = MODEL_CONFIGS["IndoBERT"]

        self.predict_btn.setEnabled(False)
        self.scrape_btn.setEnabled(False)

        self.analysis_worker = AnalysisWorker(
            [text],
            model_config
        )

        self.analysis_worker.progress.connect(
            self.on_manual_progress
        )
        self.analysis_worker.finished.connect(
            self.on_manual_prediction_finished
        )

        self.analysis_worker.start()

    def on_manual_progress(self, text):
        self.status_label.setText(text)

        current = self.manual_output.toPlainText()

        self.manual_output.setPlainText(
            current + '\n' + text if current else text
        )

    def on_manual_prediction_finished(self, results):
        self.predict_btn.setEnabled(True)
        self.scrape_btn.setEnabled(True)

        if not results:
            self.manual_output.setHtml("""
            <h2 style='color:red;'>❌ Prediksi Gagal</h2>
            """)
            return

        r = results[0]

        if r["pred_label"] == 1:
            label_text = "⚠️ SUICIDAL"
            color = "#d32f2f"
            confidence = r["prob_1"] * 100
        else:
            label_text = "✅ NON-SUICIDAL"
            color = "#2e7d32"
            confidence = r["prob_0"] * 100
    
        self.manual_output.setHtml(f"""
        <div style="
            border:2px solid {color};
            border-radius:12px;
            padding:15px;
            margin:10px;
            background:#fafafa;
        ">

            <h1 style="color:{color}; text-align:center;">
                {label_text}
            </h1>

            <hr>

            <h3>Confidence</h3>
            <p style="font-size:20px; font-weight:bold; color:{color};">
                {confidence:.2f}%
            </p>

            <h3>Input Text</h3>
            <p>{r['comment']}</p>

            <h3>Preprocessed Text</h3>
            <p style="color:#666;">
                {r['preprocessed_text']}
            </p>

        </div>
        """)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header aplikasi (judul)
        title = QLabel('X/Tweeter suicide detection')
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        layout.addWidget(title)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d8d8d8;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f2f2f2;
                color: #333333;
                min-width: 90px;
                padding: 8px 18px;
                margin-right: 2px;
                border: 1px solid #d0d0d0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #f2f2f2;
                color: #333333;
                border: 1px solid #0f62fe;
                border-bottom-color: #d8d8d8;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e8eefc;
            }
        """)

        # =========================
        # TAB MANUAL PREDICTION
        # =========================

        manual_tab = QWidget()
        manual_tab_layout = QVBoxLayout(manual_tab)

        manual_group = QGroupBox("Manual Text Prediction")
        manual_layout = QVBoxLayout()

        self.manual_text = QTextEdit()
        self.manual_text.setPlaceholderText(
            "Masukkan teks yang ingin diprediksi..."
        )

        self.predict_btn = QPushButton("Predict Text")
        self.predict_btn.clicked.connect(self.predict_manual_text)

        manual_layout.addWidget(self.manual_text)
        manual_layout.addWidget(self.predict_btn)

        manual_group.setLayout(manual_layout)

        manual_tab_layout.addWidget(manual_group)

        self.manual_output = QTextEdit()
        self.manual_output.setReadOnly(True)

        manual_tab_layout.addWidget(self.manual_output)

        self.tab_widget.addTab(
            manual_tab,
            "Manual"
        )

        # =========================
        # TAB X SCRAPING
        # =========================

        scrape_tab = QWidget()
        scrape_layout = QVBoxLayout(scrape_tab)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel('Profile URL:'))

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText('https://x.com/username')

        url_row.addWidget(self.url_edit)

        scrape_layout.addLayout(url_row)

        action_row = QHBoxLayout()

        action_row.addWidget(
            QLabel('Jumlah Tweets: recent 10 tweets')
        )

        action_row.addStretch()

        action_row.addWidget(QLabel('Model:'))

        self.model_label = QLabel('IndoBERT')
        self.model_label.setFont(
            QFont('Segoe UI', 10, QFont.Bold)
        )

        action_row.addWidget(self.model_label)

        action_row.addStretch()

        self.scrape_btn = QPushButton('Scrape Tweets')
        self.scrape_btn.clicked.connect(self.start_scrape)

        action_row.addWidget(self.scrape_btn)

        scrape_layout.addLayout(action_row)

        self.scrape_output = QTextEdit()
        self.scrape_output.setReadOnly(True)

        scrape_layout.addWidget(self.scrape_output)

        self.tab_widget.addTab(
            scrape_tab,
            "X Scraping"
        )

        csv_tab = QWidget()
        csv_layout = QVBoxLayout(csv_tab)
        
        file_row = QHBoxLayout()

        file_row.addWidget(QLabel("File"))

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)

        file_row.addWidget(self.file_edit)

        file_type_row = QHBoxLayout()

        file_type_row.addWidget(QLabel("File Type"))

        self.file_type_cb = QComboBox()
        self.file_type_cb.addItems([
            "Auto Detect",
            "CSV",
            "Excel",
            "TXT"
        ])

        file_type_row.addWidget(self.file_type_cb)

        csv_layout.addLayout(file_type_row)

        self.header_checkbox = QCheckBox(
            "First row contains header"
        )

        self.header_checkbox.setChecked(True)

        csv_layout.addWidget(
            self.header_checkbox
        )
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(
            self.browse_file
        )

        file_row.addWidget(self.browse_btn)

        csv_layout.addLayout(file_row)

        col_row = QHBoxLayout()

        col_row.addWidget(
            QLabel("Text Column")
        )

        self.text_col_cb = QComboBox()
        self.text_col_cb.setEnabled(False)

        col_row.addWidget(self.text_col_cb)

        csv_layout.addLayout(col_row)

        self.csv_predict_btn = QPushButton(
            "Predict CSV"
        )

        self.csv_predict_btn.clicked.connect(
            self.start_csv_prediction
        )

        self.csv_predict_btn.setEnabled(False)

        csv_layout.addWidget(
            self.csv_predict_btn
        )

        self.csv_output = QTextEdit()
        self.csv_output.setReadOnly(True)

        csv_layout.addWidget(
            self.csv_output
        )

        self.save_btn = QPushButton(
            "Save Results"
        )

        self.save_btn.clicked.connect(
            self.save_results
        )

        self.save_btn.setEnabled(False)

        csv_layout.addWidget(
            self.save_btn
        )
        self.tab_widget.addTab(
            csv_tab,
            "CSV / Excel"
        )

        layout.addWidget(self.tab_widget, 1)

        bottom = QHBoxLayout()
        self.status_label = QLabel('Ready')
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        layout.addLayout(bottom)

        self.setStyleSheet('''
            QWidget { background-color: #ffffff; color: #1f1f1f; }
            QLineEdit, QTextEdit {
                background-color: #ffffff;
                color: #1f1f1f;
                border: 1px solid #d9d9d9;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #0f62fe;
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold
            }
            QPushButton:hover { background-color: #0353e9; }
            QPushButton:disabled { background-color: #bdbdbd; }
        ''')

        self.comments = []
        # stats_tab = QWidget()
        # stats_layout = QVBoxLayout(stats_tab)

        # self.stats_output = QTextEdit()
        # stats_layout.addWidget(self.stats_output)

        # self.tab_widget.addTab(stats_tab, "Statistics")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv *.xlsx *.xls *.txt);;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls);;"
            "Text Files (*.txt)"
        )

        if not file_path:
            return

        try:

            self.file_edit.setText(file_path)

            # Header option
            header = 0 if self.header_checkbox.isChecked() else None

            # File type option
            selected_type = self.file_type_cb.currentText()

            # Auto detect
            if selected_type == "Auto Detect":

                ext = os.path.splitext(file_path)[1].lower()

                if ext == ".csv":
                    selected_type = "CSV"

                elif ext in [".xls", ".xlsx"]:
                    selected_type = "Excel"

                elif ext == ".txt":
                    selected_type = "TXT"

                else:
                    raise ValueError(
                        "Unsupported file type"
                    )

            # CSV
            if selected_type == "CSV":

                self.df = pd.read_csv(
                    file_path,
                    header=header
                )

            # Excel
            elif selected_type == "Excel":

                self.df = pd.read_excel(
                    file_path,
                    header=header,
                    engine="openpyxl"
                )

            # TXT
            elif selected_type == "TXT":

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    lines = [
                        line.strip()
                        for line in f
                        if line.strip()
                    ]

                self.df = pd.DataFrame({
                    "text": lines
                })

            else:
                raise ValueError(
                    "Unsupported file type"
                )

            # Kalau tidak ada header
            if header is None:

                self.df.columns = [
                    f"Column_{i}"
                    for i in range(
                        len(self.df.columns)
                    )
                ]

            columns = list(self.df.columns)

            self.text_col_cb.clear()

            for col in columns:
                self.text_col_cb.addItem(str(col))

            # otomatis pilih kolom text kalau ada
            for i, col in enumerate(columns):

                if str(col).lower() in [
                    "text",
                    "tweet",
                    "comment",
                    "content"
                ]:

                    self.text_col_cb.setCurrentIndex(i)
                    break

            self.text_col_cb.setEnabled(True)
            self.csv_predict_btn.setEnabled(True)

            self.csv_output.setPlainText(
                f"Loaded file successfully\n\n"
                f"Rows: {len(self.df)}\n"
                f"Columns: {len(self.df.columns)}\n\n"
                f"Available columns:\n"
                + "\n".join(
                    [str(c) for c in columns]
                )
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load file:\n{e}"
            )

    def start_csv_prediction(self):
        text_col = self.text_col_cb.currentText()

        texts = (
            self.df[text_col]
            .astype(str)
            .tolist()
        )

        self.csv_output.clear()

        model_config = MODEL_CONFIGS["IndoBERT"]

        self.analysis_worker = AnalysisWorker(
            texts,
            model_config
        )

        self.analysis_worker.progress.connect(
            self.on_csv_progress
        )

        self.analysis_worker.finished.connect(
            self.on_csv_finished
        )

        self.analysis_worker.start()

    def on_csv_progress(self, text):
        self.status_label.setText(text)

        current = self.csv_output.toPlainText()

        self.csv_output.setPlainText(
            current + "\n" + text
            if current else text
        )

    def on_csv_finished(self, results):
        self.prediction_results = results

        label_0 = sum(
            1 for r in results
            if r["pred_label"] == 0
        )

        label_1 = sum(
            1 for r in results
            if r["pred_label"] == 1
        )

        label_0_examples = [
            r for r in results
            if r["pred_label"] == 0
        ][:10]

        label_1_examples = [
            r for r in results
            if r["pred_label"] == 1
        ][:10]

        lines = [
            "=" * 60,
            f"Total Data : {len(results)}",
            f"Label 0 : {label_0}",
            f"Label 1 : {label_1}",
            "=" * 60,
            "",
            "CONTOH LABEL 1 (Suicidal)",
            "-" * 60,
        ]

        if label_1_examples:
            for i, r in enumerate(label_1_examples, 1):
                lines.append(
                    f"[{i}] confidence={r['prob_1']:.3f}"
                )

                lines.append(
                    r["comment"][:200]
                )

                lines.append("")
        else:
            lines.append("(kosong)")
        
        lines.extend([
            "",
            "CONTOH LABEL 0 (Non-Suicidal)",
            "-" * 60,
        ])

        if label_0_examples:
            for i, r in enumerate(label_0_examples, 1):
                lines.append(
                    f"[{i}] confidence={r['prob_0']:.3f}"
                )

                lines.append(
                    r["comment"][:200]
                )

                lines.append("")
        else:
            lines.append("(kosong)")

        self.csv_output.setPlainText(
            "\n".join(lines)
        )

        self.save_btn.setEnabled(True)
        self.csv_predict_btn.setEnabled(True)

    def save_results(self):
        if not self.prediction_results:
            QMessageBox.warning(self, 'Warning', 'No results to save')
            return
        
        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Results',
            'prediction_results.csv',
            'CSV Files (*.csv)'
        )
        
        if not save_path:
            return
        
        try:
            # Create dataframe with results
            result_df = pd.DataFrame(self.prediction_results)
            result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            QMessageBox.information(self, 'Success', f'Results saved to:\n{save_path}')
            self.status_label.setText(f'✓ Saved to {os.path.basename(save_path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save:\n{str(e)}')

    def start_scrape(self):
        # Mulai proses scraping: ambil URL dari input, jalankan ScrapeWorker
        # Hanya memicu worker; UI diupdate oleh sinyal worker (on_progress/on_finished)
        profile_url = self.url_edit.text().strip()
        
        if not profile_url:
            self.status_label.setText('⚠️ Masukkan URL profil Twitter/X!')
            return
        
        self.scrape_output.clear()
        self.tab_widget.setCurrentIndex(1)
        self.status_label.setText('Starting...')
        self.scrape_btn.setEnabled(False)
        self.profile_info = None

        self.worker = ScrapeWorker(profile_url)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, text):
        # Callback untuk update status dari worker
        # Menampilkan pesan di status bar dan juga menambahkan ke log `output`
        self.status_label.setText(text)
        current = self.scrape_output.toPlainText()
        self.scrape_output.setPlainText(
            current + '\n' + text if current else text
        )

    def on_analysis_finished(self, results):
        # Diterima setelah AnalysisWorker selesai.

        if not results:
            # Jika kosong -> tampilkan pesan gagal dan re-enable UI
            current = self.scrape_output.toPlainText()
            self.scrape_output.append('\n' + '='*80)
            self.scrape_output.append('❌ ANALYSIS GAGAL')
            self.scrape_output.append('='*80)
            self.scrape_output.append('\nLihat error detail di atas ↑↑↑\n')
            self.scrape_btn.setEnabled(True)
            self.status_label.setText('❌ Analysis failed - check output for details')
            return

        # Tampilkan hasil yang sudah dianalisis
        self.finalize_analysis_display(results, profile_info=self.profile_info)

    def finalize_analysis_display(self, results, profile_info=None):
        label_1_rows = [r for r in results if r['pred_label'] == 1]
        label_0_rows = [r for r in results if r['pred_label'] == 0]
        label_0_count = len(label_0_rows)
        label_1_count = len(label_1_rows)

        lines = [
            'LABEL 1',
            '-' * 40,
        ]
        if label_1_rows:
            for i, r in enumerate(label_1_rows, 1):
                lines.append(f"[{i}] confidence={r['prob_1']:.3f}")
                lines.append("Original:")
                lines.append(r['comment'])

                lines.append("Preprocessed:")
                lines.append(r['preprocessed_text'])

                lines.append('')
        else:
            lines.append('(kosong)')
            lines.append('')

        lines.extend([
            'LABEL 0',
            '-' * 40,
        ])
        if label_0_rows:
            for i, r in enumerate(label_0_rows, 1):
                lines.append(f"[{i}] confidence={r['prob_0']:.3f}")
                lines.append("Original:")
                lines.append(r['comment'])

                lines.append("Preprocessed:")
                lines.append(r['preprocessed_text'])

                lines.append('')
        else:
            lines.append('(kosong)')

        self.scrape_output.setPlainText('\n'.join(lines).strip())
        # Simpan info profil dan perbarui visualisasi
        self.profile_info = profile_info

        # Re-enable tombol dan update status akhir
        self.scrape_btn.setEnabled(True)
        self.status_label.setText(f'✓ Analysis done: {label_0_count} non-suicidal, {label_1_count} suicidal')


    def on_finished(self, payload):
        # Callback ketika ScrapeWorker selesai. Payload berisi tweets + profile_info
        comments = payload.get('comments', [])
        self.profile_info = payload.get('profile_info')

        self.comments = comments
        self.scrape_btn.setEnabled(True)

        if not comments:
            # Tidak ada tweet (mungkin private atau error)
            self.scrape_output.setPlainText(
                '❌ TIDAK ADA TWEET DITEMUKAN\n\n'
                '✓ Periksa Twitter API Bearer Token'
            )
            self.status_label.setText('Finished (0 tweets)')
            return

        # Tampilkan daftar tweet di tab Output
        lines = ['=' * 80, f'✓ HASIL: {len(comments)} tweet berhasil di-scrape', '=' * 80, '']
        for i, comment in enumerate(comments, 1):
            clean_comment = comment.strip().replace('\n', ' ')
            lines.append(f"[{i}] {clean_comment}")
            lines.append('-' * 80)

        self.scrape_output.setPlainText("\n".join(lines))
        self.status_label.setText(f"✓ {len(comments)} tweets scraped, starting analysis...")

        # Langsung mulai analysis worker dengan model yang ditentukan (fixed)
        model_config = MODEL_CONFIGS["IndoBERT"]
        self.scrape_btn.setEnabled(False)
        self.status_label.setText(f'⏳ Starting analysis with {"IndoBERT"}...')

        self.analysis_worker = AnalysisWorker(self.comments, model_config)
        self.analysis_worker.progress.connect(self.on_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.start()

def main():
    # Entrypoint: buat QApplication, tampilkan window utama, dan jalankan loop
    app = QApplication(sys.argv)
    win = TwitterScraperWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()