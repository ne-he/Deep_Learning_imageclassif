# Deploy NemVision

Arsitektur:

```
[ Vercel ]  frontend statis (frontend/imageclass)
     │  POST /predict  (foto)
     ▼
[ HF Space ]  FastAPI + ResNet50 (backend/)  →  JSON { label, distribution, gradcam, original }
```

Frontend statis di Vercel, model PyTorch (ResNet50, 91.6%) jalan di Hugging Face
Space. Frontend manggil API lewat `fetch`.

---

## A. Test backend di lokal (opsional, tapi disaranin)

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows PowerShell/CMD
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r backend/requirements.txt
uvicorn backend.app:app --port 7860
```

Lalu cek:

- Buka http://localhost:7860/health  → harus `{"status":"ok","model":"resnet50",...}`
- Test prediksi (ganti path ke salah satu gambar dataset):
  ```bash
  curl -F "file=@data/dataset-resized/plastic/plastic1.jpg" http://localhost:7860/predict
  ```
  Harus balikin JSON berisi `label`, `distribution`, `gradcam` (data URI), `original`.

---

## B. Deploy backend ke Hugging Face Space (Docker)

1. **Bikin Space**: huggingface.co → New Space → **SDK = Docker** → Blank → create.
2. **Clone Space-nya** (repo kosong):
   ```bash
   git clone https://huggingface.co/spaces/<username>/nemvision
   cd nemvision
   ```
3. **Copy file ini dari project ke folder Space** (struktur harus sama persis):
   ```
   nemvision/
   ├── Dockerfile                         ← copy dari backend/Dockerfile ke ROOT
   ├── README.md                          ← isi front-matter di bawah
   ├── backend/   (app.py, requirements.txt)
   ├── src/       (seluruh folder src project)
   ├── configs/   (seluruh folder configs project)
   └── outputs/resnet50/best_model.pth    ← 94MB, butuh git-LFS
   ```
   > Dockerfile WAJIB di root Space (HF baca `Dockerfile` di root). Isinya sama
   > dengan `backend/Dockerfile`, path COPY-nya udah cocok dari root.

4. **Aktifin LFS buat checkpoint 94MB**:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   ```
5. **README.md Space** (front-matter wajib biar HF tau ini Docker):
   ```
   ---
   title: NemVision Trash Classifier
   emoji: ♻️
   colorFrom: green
   colorTo: gray
   sdk: docker
   app_port: 7860
   pinned: false
   ---

   FastAPI + ResNet50 Grad-CAM inference for the NemVision trash classifier.
   ```
6. **Commit + push**:
   ```bash
   git add .
   git commit -m "NemVision ResNet50 inference API"
   git push
   ```
7. Tunggu build (liat tab "Logs" di Space). Kalau sukses, API-mu di:
   ```
   https://<username>-nemvision.hf.space
   ```
   Cek `https://<username>-nemvision.hf.space/health`.

---

## C. Sambungin frontend ke API

Di [frontend/imageclass/index.html](frontend/imageclass/index.html), cari baris:

```js
const API_BASE = "".replace(/\/$/, '');
```

Isi URL Space-nya:

```js
const API_BASE = "https://<username>-nemvision.hf.space".replace(/\/$/, '');
```

Kalau dikosongin, frontend tetep jalan tapi upload-nya mock (mode demo offline).
Tombol **Sample** selalu instan (Grad-CAM bawaan), upload user → model beneran.

---

## D. Deploy frontend ke Vercel

1. Push repo ke GitHub.
2. vercel.com → Add New Project → import repo.
3. **Root Directory = `frontend/imageclass`**, Framework = **Other**, Build Command kosong.
4. Deploy → dapet `https://nemvision.vercel.app`.

---

## E. Kunci CORS (opsional, setelah tau domain Vercel)

Default backend nerima semua origin (`*`). Biar lebih rapi, di Space →
Settings → Variables, set:

```
ALLOWED_ORIGINS = https://nemvision.vercel.app
```

---

## Catatan

- **Cold start**: Space gratis "tidur" kalau nganggur; request pertama bisa lambat
  ~30 detik buat bangun. Frontend tetap responsif (ada state "Running the model").
- **Ganti model**: backend baca env `MODEL_ARCH` (default `resnet50`). Mau coba
  mobilenet_v2? Set `MODEL_ARCH=mobilenet_v2` + upload checkpoint-nya, ganti juga
  label & angka akurasi di frontend.
