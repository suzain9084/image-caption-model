## Image Caption Model

This repository hosts a multimodal image captioning pipeline with support for EfficientNet encoders and both attention-based and GPT-2 prefix decoders. It now also includes a production-ready FastAPI backend plus a Next.js frontend so you can upload an image and receive a caption generated with the `experiments/experiment_3/checkpoint/best_model.pth` weights.

### Backend API

1. Create/activate your Python environment and install dependencies:

```
pip install -r requirements.txt
```

2. Start the API server:

```
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The server loads `config/config.yaml`, restores the experiment 3 checkpoint, and exposes:

- `GET /health` – readiness probe.
- `POST /generate-caption` – multipart form field `image`; returns `{"caption": "<text>"}`.

Example request:

```
curl -X POST http://localhost:8000/generate-caption ^
  -F "image=@sample.jpg"
```

### Frontend (Next.js)

1. Install Node.js 18+.
2. From `frontend/` install dependencies:

```
cd frontend
npm install
```

3. (Optional) configure API base URL via `frontend/.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

4. Launch the dev server:

```
npm run dev
```

Visit `http://localhost:3000`, upload an image, preview it, and click **Generate Caption** to call the backend API.

### Notes

- The backend currently supports checkpoints trained with the HF tokenizer + GPT-2 prefix decoder configuration. Ensure `experiments/experiment_3/checkpoint/best_model.pth` is present before starting the server.
- For GPU inference, install CUDA-enabled PyTorch before running `pip install -r requirements.txt`.

