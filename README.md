# RAG vs LLM-Only: PubMedQA Üzerinde Karşılaştırmalı Değerlendirme

Bu proje, Retrieval-Augmented Generation (RAG) ve sadece LLM (Large Language Model) yaklaşımlarını PubMedQA veri seti üzerinde karşılaştırmalı olarak değerlendirmektedir. DPR (Dense Passage Retrieval) ve SBERT (Sentence-BERT) tabanlı RAG sistemleri ile çeşitli LLM modellerinin performansını F1 skoru, doğruluk ve güvenilirlik (faithfulness) metrikleri ile ölçmektedir. Ayrıca detaylı görselleştirme ve analiz araçları içermektedir.

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)
- [Teknik Detaylar](#teknik-detaylar)
- [Yeni Özellikler](#yeni-özellikler)
- [Gelecek İyileştirmeler](#gelecek-iyileştirmeler)

## Proje Özeti

Bu proje, biyomedikal soru-cevap görevinde RAG sisteminin performansını değerlendirmek için tasarlanmıştır. Farklı yaklaşımlar karşılaştırılmaktadır:

1. **LLM-Only**: LLM modellerinin sadece kendi bilgisiyle cevap üretmesi
2. **DPR-RAG**: DPR (Dense Passage Retrieval) ile belge erişimi yapıp, bu belgeleri kullanarak cevap üretmesi
3. **SBERT-RAG**: SBERT (Sentence-BERT) ile belge erişimi yapıp, bu belgeleri kullanarak cevap üretmesi

### Kullanılan Teknolojiler

- **LLM Modelleri**: 
  - Mistral-7B-Instruct-v0.2
  - Phi-3-mini-4k-instruct
- **Retrieval Modelleri**: 
  - **DPR (Dense Passage Retrieval)**: Facebook DPR
    - Question Encoder: `facebook/dpr-question_encoder-single-nq-base`
    - Context Encoder: `facebook/dpr-ctx_encoder-single-nq-base`
  - **SBERT (Sentence-BERT)**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Veri Seti**: PubMedQA (örnek sayısı/alt-küme seçimi deney ayarlarına göre değişebilir)
- **Görselleştirme**: Matplotlib, Seaborn

## Mimari

### RAG Pipeline

**DPR-RAG:**
```
Soru → DPR Question Encoder → Query Embedding
                                    ↓
                            FAISS Index (DPR Context Embeddings)
                                    ↓
                            Top-K Document Retrieval
                                    ↓
                    Context + Question → LLM → Cevap
```

**SBERT-RAG:**
```
Soru → SBERT Encoder → Query Embedding
                            ↓
                    FAISS Index (SBERT Embeddings)
                            ↓
                    Top-K Document Retrieval
                            ↓
            Context + Question → LLM → Cevap
```

### LLM-Only Pipeline

```
Soru → Mistral-7B → Cevap
```

## 🔧 Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (önerilir, CPU da çalışır)
- ~15GB disk alanı (modeller için)

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd bkt
```

2. **Sanal ortam oluşturun ve aktifleştirin:**
```bash
python -m venv rag543
# Windows
rag543\Scripts\activate
# Linux/Mac
source rag543/bin/activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Ek bağımlılıklar (eğer eksikse):**
```bash
pip install datasets transformers torch faiss-cpu scikit-learn tqdm sentence-transformers matplotlib seaborn
```

## 📖 Kullanım

### Tek Komutla Uçtan Uca Çalıştırma (Opsiyonel)

Tüm adımları (indirme → hazırlama → index → deney → değerlendirme → plot) tek seferde çalıştırmak için:

```bash
bash run_project_pipeline.sh
```

Notlar:
- `run_project_pipeline.sh` **bash** gerektirir. Windows’ta **WSL** veya **Git Bash** ile çalıştırın.
- Çalışma log’u kök dizinde `project_execution.log` olarak oluşur.

### 1. Veri Setini İndirme

PubMedQA veri setini indirin:
```bash
python scripts/download_pubmedqa.py
```

### 2. Corpus Hazırlama

PubMedQA context'lerini işlenebilir TSV formatına dönüştürün:
```bash
python scripts/prepare_pubmed_corpus.py
```

Eğer corpus ID'lerinde sorun varsa:
```bash
python scripts/fix_corpus_ids.py
```

### 3. Embedding ve Index Oluşturma

**DPR Index Oluşturma:**
DPR Context Encoder ile corpus'u encode edin ve FAISS index'i oluşturun:
```bash
python scripts/index.py
```

Bu işlem şunları oluşturur:
- `dpr_ctx_embeddings.npy`: DPR context embedding'leri (kök dizinde)
- `dpr_faiss.index`: DPR FAISS index dosyası (kök dizinde)

**SBERT Index Oluşturma (Opsiyonel):**
SBERT modeli ile corpus'u encode edin ve FAISS index'i oluşturun:
```bash
python scripts/build_sbert_index.py
```

Bu işlem şunları oluşturur:
- `sbert_faiss.index`: SBERT tabanlı FAISS index dosyası (kök dizinde)

### 4. Deneyleri Çalıştırma

LLM-only ve RAG deneylerini çalıştırın:
```bash
python scripts/run_experiments.py
```

Bu script şunları oluşturur:
- `llm_only_<LLM>.jsonl`: LLM-only sonuçları (örn. `llm_only_Mistral-7B.jsonl`, `llm_only_Phi-3.jsonl`)
- `rag_<LLM>_<Retriever>.jsonl`: RAG sonuçları (örn. `rag_Mistral-7B_DPR.jsonl`, `rag_Phi-3_SBERT.jsonl`)

### 5. Değerlendirme

#### F1 Skoru ve Doğruluk Değerlendirmesi
```bash
python scripts/evaluate_f1.py
```

#### Güvenilirlik (Faithfulness) Değerlendirmesi
```bash
python scripts/evaluate_faithfulness.py
```

#### Görselleştirme ve Analiz
Sonuçları görselleştirmek ve detaylı analiz yapmak için:
```bash
python scripts/analysis_plots.py
```

Bu script şu görselleştirmeleri oluşturur:
- LLM-Only ve RAG için confusion matrix'ler
- Soru bazlı doğruluk karşılaştırması
- Retrieval skor dağılımı
- Güvenilirlik vs Doğruluk scatter plot'u
Ve çıktıları `results/` klasörüne kaydeder.

### Tekil Test

Tek bir soru için test yapmak isterseniz:

**LLM-Only:**
```bash
python scripts/run_llm_only.py
```

**RAG (DPR):**
```bash
python scripts/run_rag.py
```

**SBERT Retriever Test:**
SBERT retriever'ı doğrudan test etmek için:
```bash
python scripts/retrieve.py
```

Bu script SBERT retriever'ı kullanarak örnek bir sorgu çalıştırır.

## Sonuçlar

### F1 Skoru ve Doğruluk Metrikleri

En güncel metrikler `results/metrics_summary.csv` ve özet rapor `results/summary_report.txt` içinde bulunur.

Aşağıdaki tablo, mevcut `results/metrics_summary.csv` çıktısındaki değerleri özetler:

| Model | Samples | Macro-F1 | Accuracy |
|------|---------:|---------:|---------:|
| LLM-only (Mistral-7B) | 1000 | 0.2756 | 0.326 |
| LLM-only (Phi-3) | 1000 | 0.1822 | 0.181 |
| RAG (Mistral-7B + DPR) | 1000 | 0.2524 | 0.315 |
| RAG (Mistral-7B + SBERT) | 1000 | 0.2698 | 0.308 |
| RAG (Phi-3 + DPR) | 1000 | 0.3151 | 0.367 |
| RAG (Phi-3 + SBERT) | 1000 | 0.3128 | 0.371 |

### Güvenilirlik (Faithfulness) Metrikleri

Faithfulness metrikleri sadece RAG koşulları için hesaplanır. En güncel özet `results/summary_report.txt` ve `results/faithfulness_summary.csv` dosyalarındadır.

### Sonuç Analizi

`results/summary_report.txt` özetine göre:

1. **En iyi Accuracy**: 0.3710 (RAG (Phi-3 + SBERT))
2. **En iyi Macro-F1**: 0.3151 (RAG (Phi-3 + DPR))
3. **Genel gözlem**: RAG, LLM-only’e göre bazı karşılaştırmalarda daha iyi sonuç verir; kesin hüküm için aynı LLM ile koşul bazında kıyas yapmak gerekir.

### Detaylı Batch Sonuç Analizi

#### 1. Sınıf Bazlı Performans Analizi

**LLM-Only Model:**
- Model, çoğu durumda "yes" cevabı verme eğilimindedir
- "maybe" ve "no" sınıflarını ayırt etmekte zorlanmaktadır
- Genel olarak daha uzun ve detaylı cevaplar üretmektedir (ortalama ~500-800 karakter)
- Cevap üretiminde kendi eğitim bilgisini kullanmaktadır, bu da bazen yanlış bilgilere yol açabilmektedir

**DPR-RAG Model:**
- Retrieve edilen belgelere dayalı cevaplar üretmektedir
- "Insufficient evidence" ifadesini sıkça kullanmaktadır (güvenilirlik açısından olumlu)
- Cevap uzunlukları daha kısa ve odaklıdır (ortalama ~200-400 karakter)
- Retrieve edilen belgelerin kalitesi cevap kalitesini doğrudan etkilemektedir

#### 2. Retrieval Kalitesi Analizi

**Retrieval Skorları:**
- En yüksek similarity skoru: ~0.69 (mitochondria ve PCD sorusu)
- En düşük similarity skoru: ~0.59 (bazı genel tıbbi sorular)
- Ortalama similarity skoru: ~0.64
- Skorlar 0.59-0.69 aralığında dağılmıştır, bu da retrieval sisteminin tutarlı çalıştığını göstermektedir

**Retrieval Başarısı:**
- Top-1 belge genellikle soruyla ilgili içerik sağlamaktadır
- Ancak bazı durumlarda retrieve edilen belgeler soruyu tam olarak yanıtlamamaktadır
- Bu durumda model "Insufficient evidence" cevabı vermektedir

#### 3. Cevap Üretim Kalitesi Karşılaştırması

**LLM-Only Özellikleri:**
- Daha akıcı ve doğal dil kullanımı
- Detaylı açıklamalar ve bağlam sağlama
- Bazen yanlış veya güncel olmayan bilgiler
- Hallucination riski yüksek
- PubMedQA formatına uyum sorunu (yes/no/maybe)

**DPR-RAG Özellikleri:**
- Retrieve edilen belgelere dayalı, daha güvenilir cevaplar
- "Insufficient evidence" kullanımı ile şeffaflık
- Daha kısa ve odaklı cevaplar
- Bazen retrieve edilen belgeler yetersiz kalabilmektedir
- Cevap üretiminde daha mekanik bir dil kullanımı

#### 4. Hata Analizi

**LLM-Only Hataları:**
1. **Sınıf Yanlış Sınıflandırması**: "yes" sorusuna "no" veya "maybe" cevabı verme
2. **Aşırı Genelleme**: Model kendi bilgisini kullanarak kesin cevaplar verme eğilimi
3. **Format Uyumsuzluğu**: Uzun açıklamalar yerine kısa yes/no/maybe bekleniyor

**DPR-RAG Hataları:**
1. **Retrieval Başarısızlığı**: İlgili belgelerin retrieve edilememesi
2. **Yetersiz Context**: Retrieve edilen belgelerin soruyu tam yanıtlayamaması
3. **Aşırı İhtiyatlılık**: Yeterli bilgi olsa bile "Insufficient evidence" verme

#### 5. Başarılı Örnekler

**RAG Başarılı Örnekler:**
- **Soru**: "30-Day and 1-year mortality in emergency general surgery laparotomies..."
- **Sonuç**: Retrieve edilen belgeler doğrudan ilgili istatistikleri içermekte, model doğru cevap üretebilmektedir
- **Retrieval Skoru**: 0.69 (yüksek)

**LLM-Only Başarılı Örnekler:**
- **Soru**: "Do mitochondria play a role in remodelling lace plant leaves..."
- **Sonuç**: Model genel bilgisiyle doğru yönde cevap verebilmektedir
- **Not**: Ancak bu tür başarılar tutarsızdır

#### 6. İyileştirme Önerileri

**Retrieval İyileştirmeleri:**
- Top-K değerini artırmak (5 → 10)
- Query expansion teknikleri kullanmak
- Hybrid retrieval (dense + sparse) denemek
- Re-ranking mekanizması eklemek

**Generation İyileştirmeleri:**
- Prompt engineering ile format uyumunu artırmak
- Few-shot örnekler eklemek
- Temperature ve sampling parametrelerini optimize etmek
- Post-processing ile cevap normalizasyonu

**Sistem İyileştirmeleri:**
- Daha büyük corpus kullanmak
- Domain-specific fine-tuning
- Ensemble yöntemleri
- Active learning ile veri toplama

#### 7. İstatistiksel Özet

**Cevap Uzunlukları:**
- **LLM-Only**: Ortalama ~650 karakter, aralık: 200-1200 karakter
- **DPR-RAG**: Ortalama ~350 karakter, aralık: 100-800 karakter
- RAG cevapları daha kısa ve odaklıdır

**"Insufficient Evidence" Kullanımı:**
- RAG modeli, retrieve edilen belgelerin yetersiz olduğu durumlarda şeffaflık sağlamak için bu ifadeyi kullanmaktadır
- Bu yaklaşım, güvenilirlik açısından olumlu bir özelliktir
- LLM-only model bu ifadeyi kullanmamaktadır

**Retrieval Dağılımı:**
- Top-1 belge skorları: 0.59 - 0.69 aralığında
- Skorların çoğu 0.62-0.67 aralığında yoğunlaşmıştır
- Bu, retrieval sisteminin tutarlı çalıştığını göstermektedir

**Sınıf Dağılımı (Gold Labels):**
- Bu dağılım, seçilen örnek sayısına ve kullanılan alt-kümeye göre değişir.
- En doğru özet için deney çıktılarını (`llm_only_*.jsonl`, `rag_*.jsonl`) ve raporu (`results/summary_report.txt`) referans alın.

#### 8. Performans Karşılaştırma Tablosu

Bu bölümdeki metrikleri statik yazmak yerine, `results/metrics_summary.csv` ve `results/summary_report.txt` dosyalarını kaynak olarak kullanın (deney ayarları değiştikçe değerler güncellenir).

#### 9. Örnek Çıktılar

**Örnek 1: Başarılı RAG Cevabı**

**Soru**: "30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement?"

**Retrieved Documents**: Emergency surgery mortality istatistikleri içeren belgeler (skor: 0.69)

**RAG Cevabı**: 
> "Based on the context, the 30-day mortality was 12.4% for patients under 70 years and 22% for those over 70 years, and the 1-year mortality was 25.1% for patients under 70 years and 38% for those over 70 years. These findings suggest that there is a higher mortality rate for older patients both in the short and long term. Therefore, it appears that there is a need for improvement in the area of emergency general surgery laparotomies, particularly for older patients."

**Değerlendirme**:  Retrieve edilen belgeler doğrudan ilgili, cevap belgelere dayalı

---

**Örnek 2: LLM-Only vs RAG Karşılaştırması**

**Soru**: "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"

**LLM-Only Cevabı**:
> "Mitochondria do play a role in programmed cell death (PCD) in plants, including the remodeling of lace plant leaves. During PCD, mitochondria undergo various changes, such as swelling, membrane potential loss, and the release of reactive oxygen species (ROS) and other pro-apoptotic factors..."

**RAG Cevabı**:
> "Based on the context, there is evidence that mitochondria play a role in programmed cell death (PCD) in the lace plant (Aponogeton madagascariensis). The study examines mitochondrial dynamics during PCD and shows that mitochondrial permeability transition pore (PTP) formation may be involved. However, the exact role of mitochondria in remodelling the lace plant leaves during PCD is not explicitly stated in the context. Therefore, the answer is: Insufficient evidence."

**Değerlendirme**: 
- LLM-Only: Genel bilgi kullanarak kesin cevap veriyor (hallucination riski)
- RAG: Retrieve edilen belgelere dayalı, şeffaf ve güvenilir cevap

#### 10. Sonuç ve Öneriler

**Ana Bulgular:**
1. Retrieval (DPR/SBERT) seçimi ve kullanılan LLM, performansı belirgin şekilde etkiler.
2. RAG için faithfulness metrikleri `results/faithfulness_summary.csv` içinde özetlenir.
3. Retrieval kalitesi, RAG performansını doğrudan etkiler.

**Pratik Öneriler:**
- Biyomedikal QA görevleri için RAG yaklaşımı önerilir
- Retrieval sisteminin optimize edilmesi kritik öneme sahiptir
- Prompt engineering ile format uyumu artırılabilir
- Daha büyük ve kaliteli corpus kullanımı performansı artıracaktır

##  Proje Yapısı

```
bkt/
├── data/
│   ├── raw/
│   │   └── pubmedqa.json              # Ham PubMedQA veri seti
│   └── processed/
│       ├── pubmed_corpus.tsv           # İşlenmiş corpus
│       └── pubmed_corpus_fixed.tsv     # ID'leri düzeltilmiş corpus
├── scripts/
│   ├── download_pubmedqa.py           # Veri seti indirme
│   ├── prepare_pubmed_corpus.py       # Corpus hazırlama
│   ├── fix_corpus_ids.py              # Corpus ID düzeltme
│   ├── index.py                       # DPR FAISS index oluşturma
│   ├── build_sbert_index.py           # SBERT FAISS index oluşturma
│   ├── retrieve.py                    # DPR ve SBERT ile belge erişimi
│   ├── generate.py                    # LLM ile cevap üretme
│   ├── run_experiments.py             # Toplu deney çalıştırma
│   ├── run_llm_only.py                # LLM-only test
│   ├── run_rag.py                     # RAG test
│   ├── evaluate_f1.py                 # F1 ve doğruluk değerlendirme
│   ├── evaluate_faithfulness.py       # Güvenilirlik değerlendirme
│   └── analysis_plots.py              # Görselleştirme ve analiz
├── results/                           # Raporlar, CSV özetler ve görseller
│   ├── metrics_summary.csv
│   ├── faithfulness_summary.csv
│   ├── summary_report.txt
│   └── *.png
├── dpr_ctx_embeddings.npy             # DPR context embedding'leri (kök dizin)
├── dpr_faiss.index                    # DPR FAISS index (kök dizin)
├── sbert_faiss.index                  # SBERT FAISS index (kök dizin)
├── llm_only_*.jsonl                   # LLM-only sonuçları (kök dizin)
├── rag_*.jsonl                        # RAG sonuçları (kök dizin)
├── requirements.txt                   # Python bağımlılıkları
├── run_project_pipeline.sh            # Uçtan uca pipeline (bash)
└── README.md                          # Bu dosya
```

## 🔬 Teknik Detaylar

### Model Konfigürasyonu

- **LLM Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Device**: CUDA (varsa), aksi halde CPU
- **Dtype**: float16 (CUDA), float32 (CPU)
- **Max Tokens**: 128 (varsayılan)

### Retrieval Konfigürasyonu

**DPR:**
- **Top-K**: 5 (retrieve edilen belge sayısı)
- **Embedding Dimension**: 768 (DPR base model)
- **Similarity Metric**: Cosine Similarity (Inner Product)
- **Max Length**: 512 token

**SBERT:**
- **Top-K**: 5 (retrieve edilen belge sayısı)
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Similarity Metric**: Cosine Similarity (Inner Product)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Prompt Tasarımı

**RAG Modu:**
```
You are a biomedical question answering assistant.

Answer the question ONLY using the information provided in the context.
Do NOT use any external knowledge.
If the answer cannot be derived from the context, say "Insufficient evidence."

Context:
[Retrieved documents]

Question:
[Question]

Answer:
```

**LLM-Only Modu:**
```
You are a biomedical question answering assistant.

Answer the following question to the best of your knowledge.

Question:
[Question]

Answer:
```

### Değerlendirme Metrikleri

1. **Macro-F1**: Tüm sınıflar (yes/no/maybe) için ortalama F1 skoru
2. **Accuracy**: Doğru tahmin yüzdesi
3. **Citation Overlap**: Cevaptaki token'ların retrieve edilen belgelerdeki token'larla örtüşme oranı
4. **Attribution Score**: Cevaptaki cümlelerin retrieve edilen belgeler tarafından desteklenme oranı

##  Yeni Özellikler

### SBERT (Sentence-BERT) Desteği
- SBERT tabanlı alternatif retrieval sistemi eklendi
- DPR'ye göre daha hızlı ve hafif bir alternatif
- `build_sbert_index.py` scripti ile SBERT index'i oluşturulabilir
- `SBERTRetriever` sınıfı ile kullanılabilir

### Görselleştirme ve Analiz
- `analysis_plots.py` scripti ile detaylı görselleştirmeler
- Confusion matrix'ler (LLM-Only ve RAG)
- Soru bazlı doğruluk karşılaştırması
- Retrieval skor dağılımı analizi
- Güvenilirlik vs doğruluk korelasyon analizi

### Çoklu LLM Desteği
- Mistral-7B-Instruct
- Phi-3-mini-4k-instruct
- `run_experiments.py` ile farklı LLM'ler test edilebilir

##  Gelecek İyileştirmeler

- [x] SBERT retrieval desteği
- [x] Görselleştirme ve analiz araçları
- [x] Çoklu LLM desteği
- [ ] Daha büyük örnek seti ile deneyler (50 → 500+)
- [ ] Hybrid retrieval (DPR + SBERT) denemeleri
- [ ] Re-ranking mekanizması eklenmesi

##  Notlar

- İlk çalıştırmada modeller otomatik olarak indirilecektir (~15GB)
- GPU kullanımı önerilir, ancak CPU ile de çalışır (daha yavaş)
- FAISS index oluşturma işlemi corpus boyutuna göre zaman alabilir

### Çıktılar Nerede?

- **Veri**: `data/raw/` ve `data/processed/`
- **Index dosyaları**: kök dizinde `dpr_faiss.index`, `sbert_faiss.index` (ve `dpr_ctx_embeddings.npy`)
- **Deney çıktı dosyaları**: kök dizinde `llm_only_*.jsonl`, `rag_*.jsonl`
- **Rapor/plot**: `results/`

##  Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

##  Yazar

Term Project - RAG vs LLM-Only Karşılaştırması

---

**Son Güncelleme**: 2024

