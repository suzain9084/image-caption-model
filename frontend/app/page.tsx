"use client";

import { ChangeEvent, useEffect, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    setCaption(null);
    setError(null);
    if (!nextFile) {
      setFile(null);
      return;
    }

    if (!nextFile.type.startsWith("image/")) {
      setError("Please choose an image file.");
      setFile(null);
      return;
    }

    setFile(nextFile);
  };

  const handleGenerate = async () => {
    if (!file) {
      setError("Select an image before generating a caption.");
      return;
    }

    setIsLoading(true);
    setCaption(null);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`${API_BASE}/generate-caption`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Failed to generate caption.");
      }

      const data = await response.json();
      setCaption(data.caption ?? "No caption returned.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="panel">
        <header>
          <h1>Image Caption Generator</h1>
          <p>Upload an image, preview it, and let the model describe it.</p>
        </header>

        <div className="upload">
          <input
            id="image-input"
            type="file"
            accept="image/*"
            onChange={onFileChange}
          />
          {previewUrl && (
            <div className="preview">
              <img src={previewUrl} alt="Preview" />
            </div>
          )}
        </div>

        <button
          className="action"
          onClick={handleGenerate}
          disabled={!file || isLoading}
        >
          {isLoading ? "Generating..." : "Generate Caption"}
        </button>

        {caption && (
          <div className="caption success">
            <h2>Caption</h2>
            <p>{caption}</p>
          </div>
        )}

        {error && (
          <div className="caption error">
            <h2>Something went wrong</h2>
            <p>{error}</p>
          </div>
        )}
      </section>
    </main>
  );
}

