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
    <main className="min-h-screen flex items-center justify-center p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-4xl">
        <div className="bg-slate-800/90 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-6 sm:p-8 lg:p-10 shadow-2xl shadow-black/50">
          {/* Header */}
          <header className="mb-8 text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-3">
              Image Caption Generator
            </h1>
            <p className="text-slate-400 text-lg sm:text-xl">
              Upload an image and let AI describe it for you
            </p>
          </header>

          {/* Upload Section */}
          <div className="mb-6">
            <label
              htmlFor="image-input"
              className="flex flex-col items-center justify-center w-full h-48 sm:h-64 border-2 border-dashed border-slate-600 rounded-2xl bg-slate-900/50 hover:border-cyan-500/50 hover:bg-slate-900/70 transition-all duration-300 cursor-pointer group"
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6 px-4">
                <svg
                  className="w-12 h-12 mb-4 text-slate-500 group-hover:text-cyan-400 transition-colors"
                  aria-hidden="true"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 20 16"
                >
                  <path
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
                  />
                </svg>
                <p className="mb-2 text-sm sm:text-base text-slate-400 group-hover:text-slate-300 transition-colors">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs sm:text-sm text-slate-500">
                  PNG, JPG, GIF or WEBP (MAX. 10MB)
                </p>
              </div>
              <input
                id="image-input"
                type="file"
                accept="image/*"
                onChange={onFileChange}
                className="hidden"
              />
            </label>

            {/* Image Preview */}
            {previewUrl && (
              <div className="mt-6 rounded-2xl overflow-hidden border border-slate-700/50 shadow-lg">
                <div className="relative group">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-auto max-h-96 object-contain bg-slate-900/50"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </div>
            )}
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={!file || isLoading}
            className="w-full py-4 px-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed text-white font-semibold text-lg rounded-xl shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 disabled:shadow-none transition-all duration-300 transform hover:scale-[1.02] disabled:transform-none active:scale-[0.98] flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                <span>Generating Caption...</span>
              </>
            ) : (
              <span>Generate Caption</span>
            )}
          </button>

          {/* Caption Result */}
          {caption && (
            <div className="mt-6 p-6 rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/30 shadow-lg backdrop-blur-sm transition-all duration-300">
              <div className="flex items-center gap-2 mb-3">
                <svg
                  className="w-6 h-6 text-emerald-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <h2 className="text-xl font-bold text-emerald-300">Generated Caption</h2>
              </div>
              <p className="text-slate-200 text-lg leading-relaxed">{caption}</p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-6 rounded-2xl bg-gradient-to-br from-red-500/10 to-rose-500/10 border border-red-500/30 shadow-lg backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-3">
                <svg
                  className="w-6 h-6 text-red-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <h2 className="text-xl font-bold text-red-300">Error</h2>
              </div>
              <p className="text-slate-200 text-lg">{error}</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
