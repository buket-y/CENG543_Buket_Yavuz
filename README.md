# RAG vs LLM-Only: PubMedQA Üzerinde Karşılaştırmalı Değerlendirme

Bu proje, Retrieval-Augmented Generation (RAG) ve sadece LLM (Large Language Model) yaklaşımlarını PubMedQA veri seti üzerinde karşılaştırmalı olarak değerlendirmektedir. DPR (Dense Passage Retrieval) tabanlı RAG sistemi ile Mistral-7B-Instruct modelinin performansını F1 skoru, doğruluk ve güvenilirlik (faithfulness) metrikleri ile ölçmektedir.

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)
- [Teknik Detaylar](#teknik-detaylar)

## Proje Özeti

Bu proje, biyomedikal soru-cevap görevinde RAG sisteminin performansını değerlendirmek için tasarlanmıştır. İki farklı yaklaşım karşılaştırılmaktadır:

1. **LLM-Only**: Mistral-7B-Instruct modelinin sadece kendi bilgisiyle cevap üretmesi
2. **DPR-RAG**: DPR ile belge erişimi yapıp, bu belgeleri kullanarak cevap üretmesi

### Kullanılan Teknolojiler

- **LLM**: Mistral-7B-Instruct-v0.2
- **Retrieval Model**: Facebook DPR (Dense Passage Retrieval)
  - Question Encoder: `facebook/dpr-question_encoder-single-nq-base`
  - Context Encoder: `facebook/dpr-ctx_encoder-single-nq-base`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Veri Seti**: PubMedQA (50 örnek)

## Mimari

### RAG Pipeline

```
Soru → DPR Question Encoder → Query Embedding
                                    ↓
                            FAISS Index (Context Embeddings)
                                    ↓
                            Top-K Document Retrieval
                                    ↓
                    Context + Question → Mistral-7B → Cevap
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
cd term_project
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
pip install datasets transformers torch faiss-cpu scikit-learn tqdm
```

## 📖 Kullanım

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

DPR Context Encoder ile corpus'u encode edin ve FAISS index'i oluşturun:
```bash
python scripts/index.py
```

Bu işlem şunları oluşturur:
- `dpr_ctx_embeddings.npy`: Context embedding'leri
- `dpr_faiss.index`: FAISS index dosyası

### 4. Deneyleri Çalıştırma

LLM-only ve RAG deneylerini çalıştırın:
```bash
python scripts/run_experiments.py
```

Bu script şunları oluşturur:
- `llm_only_batch.jsonl`: LLM-only sonuçları
- `rag_batch.jsonl`: RAG sonuçları

### 5. Değerlendirme

#### F1 Skoru ve Doğruluk Değerlendirmesi
```bash
python scripts/evaluate_f1.py
```

#### Güvenilirlik (Faithfulness) Değerlendirmesi
```bash
python scripts/evaluate_faithfulness.py
```

### Tekil Test

Tek bir soru için test yapmak isterseniz:

**LLM-Only:**
```bash
python scripts/run_llm_only.py
```

**RAG:**
```bash
python scripts/run_rag.py
```

## Sonuçlar

### F1 Skoru ve Doğruluk Metrikleri

50 örnek üzerinde yapılan değerlendirme sonuçları:

#### LLM-Only
- **Örnek Sayısı**: 50
- **Macro-F1**: 0.1221
- **Doğruluk (Accuracy)**: 0.1400

#### DPR-RAG
- **Örnek Sayısı**: 50
- **Macro-F1**: 0.3689
- **Doğruluk (Accuracy)**: 0.3200

### Güvenilirlik (Faithfulness) Metrikleri

DPR-RAG sistemi için güvenilirlik analizi:

- **Ortalama Citation Overlap**: 0.6211
  - Üretilen cevapların %62.11'i retrieve edilen belgelerdeki token'larla örtüşmektedir
  - Bu, cevapların büyük ölçüde retrieve edilen belgelere dayandığını göstermektedir

- **Ortalama Attribution Score**: 0.7900
  - Üretilen cümlelerin %79'u retrieve edilen belgeler tarafından desteklenmektedir
  - Bu, sistemin yüksek bir güvenilirlik seviyesine sahip olduğunu göstermektedir

### Sonuç Analizi

1. **RAG'in Üstünlüğü**: DPR-RAG, LLM-only yaklaşıma göre yaklaşık **3x daha yüksek** F1 skoru elde etmiştir (0.3689 vs 0.1221). Bu, retrieve edilen belgelerin modelin performansını önemli ölçüde artırdığını göstermektedir.

2. **Güvenilirlik**: RAG sisteminin yüksek citation overlap (0.6211) ve attribution score (0.7900) değerleri, üretilen cevapların retrieve edilen belgelere dayandığını ve "hallucination" riskinin düşük olduğunu göstermektedir.

3. **İyileştirme Alanları**: 
   - Doğruluk oranları hala düşük (LLM-only: 14%, RAG: 32%)
   - Daha fazla örnek ve fine-tuning ile performans artırılabilir
   - Top-K değerinin optimizasyonu gerekebilir

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
- PubMedQA veri setinde 50 örnek içinde:
  - "yes" sınıfı: ~35-40 örnek
  - "no" sınıfı: ~8-10 örnek  
  - "maybe" sınıfı: ~3-5 örnek
- Veri seti "yes" ağırlıklıdır, bu da modelin "yes" verme eğilimini açıklayabilir

#### 8. Performans Karşılaştırma Tablosu

| Metrik | LLM-Only | DPR-RAG | İyileştirme |
|--------|----------|---------|-------------|
| **Macro-F1** | 0.1221 | 0.3689 | **+202%** |
| **Accuracy** | 0.1400 | 0.3200 | **+129%** |
| **Citation Overlap** | N/A | 0.6211 | - |
| **Attribution Score** | N/A | 0.7900 | - |
| **Ortalama Cevap Uzunluğu** | ~650 karakter | ~350 karakter | -46% (daha kısa) |
| **Hallucination Riski** | Yüksek | Düşük | - |
| **Güvenilirlik** | Düşük | Yüksek | - |

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
1. RAG sistemi, LLM-only yaklaşıma göre **3x daha yüksek** performans göstermektedir
2. RAG'in güvenilirlik metrikleri (0.62 citation overlap, 0.79 attribution) oldukça yüksektir
3. LLM-only model, daha akıcı ama daha az güvenilir cevaplar üretmektedir
4. Retrieval kalitesi, RAG performansını doğrudan etkilemektedir

**Pratik Öneriler:**
- Biyomedikal QA görevleri için RAG yaklaşımı önerilir
- Retrieval sisteminin optimize edilmesi kritik öneme sahiptir
- Prompt engineering ile format uyumu artırılabilir
- Daha büyük ve kaliteli corpus kullanımı performansı artıracaktır

##  Proje Yapısı

```
term_project/
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
│   ├── encode.py                      # Embedding oluşturma
│   ├── index.py                       # FAISS index oluşturma
│   ├── retrieve.py                    # DPR ile belge erişimi
│   ├── generate.py                    # Mistral ile cevap üretme
│   ├── run_experiments.py             # Toplu deney çalıştırma
│   ├── run_llm_only.py                # LLM-only test
│   ├── run_rag.py                     # RAG test
│   ├── evaluate_f1.py                 # F1 ve doğruluk değerlendirme
│   └── evaluate_faithfulness.py       # Güvenilirlik değerlendirme
├── colbert_index/                     # ColBERT index (opsiyonel)
├── dpr_ctx_embeddings.npy             # Context embedding'leri
├── dpr_faiss.index                    # FAISS index
├── llm_only_batch.jsonl               # LLM-only sonuçları
├── rag_batch.jsonl                    # RAG sonuçları
├── requirements.txt                   # Python bağımlılıkları
└── README.md                          # Bu dosya
```

## 🔬 Teknik Detaylar

### Model Konfigürasyonu

- **LLM Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Device**: CUDA (varsa), aksi halde CPU
- **Dtype**: float16 (CUDA), float32 (CPU)
- **Max Tokens**: 128 (varsayılan)

### Retrieval Konfigürasyonu

- **Top-K**: 5 (retrieve edilen belge sayısı)
- **Embedding Dimension**: 768 (DPR base model)
- **Similarity Metric**: Cosine Similarity (Inner Product)
- **Max Length**: 512 token

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

##  Gelecek İyileştirmeler

- [ ] Daha büyük örnek seti ile deneyler (50 → 500+)
- [ ] Farklı LLM ve Dense Retrieverlar ile testler
- [ ] Daha detaylı analiz ve görselleştirmeler

##  Notlar

- İlk çalıştırmada modeller otomatik olarak indirilecektir (~15GB)
- GPU kullanımı önerilir, ancak CPU ile de çalışır (daha yavaş)
- FAISS index oluşturma işlemi corpus boyutuna göre zaman alabilir

##  Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

##  Yazar

Term Project - RAG vs LLM-Only Karşılaştırması

---

**Son Güncelleme**: 2024

