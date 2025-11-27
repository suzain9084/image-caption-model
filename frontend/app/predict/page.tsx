"use client";

import { ChangeEvent, useEffect, useState, useRef } from "react";
import Link from "next/link";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// Typing animation component
function TypingAnimation({ text, onComplete }: { text: string; onComplete?: () => void }) {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(text.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 30);

      return () => clearTimeout(timer);
    } else if (onComplete) {
      onComplete();
    }
  }, [currentIndex, text, onComplete]);

  useEffect(() => {
    setDisplayedText("");
    setCurrentIndex(0);
  }, [text]);

  return (
    <span>
      {displayedText}
      {currentIndex < text.length && (
        <span className="animate-pulse">|</span>
      )}
    </span>
  );
}

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showUploadArea, setShowUploadArea] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    if (file && previewUrl) {
      const timer = setTimeout(() => {
        setShowUploadArea(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [file, previewUrl]);

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

    if (!nextFile.type.match(/image\/(jpeg|png)/)) {
      setError("Only JPEG and PNG images are supported.");
      setFile(null);
      return;
    }

    setFile(nextFile);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      if (!droppedFile.type.match(/image\/(jpeg|png)/)) {
        setError("Only JPEG and PNG images are supported.");
        return;
      }
      setFile(droppedFile);
      setCaption(null);
      setError(null);
    }
  };

  const handleCancel = () => {
    setFile(null);
    setCaption(null);
    setError(null);
    setShowUploadArea(true);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
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
    <div className="h-screen bg-white flex flex-col">
      {/* Header */}
      <header className="w-full px-24 py-4 flex items-center justify-between border-b border-gray-200">
        <Link href="/" className="flex items-center gap-2">
          <svg
            className="w-6 h-6 text-gray-800"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <circle
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="2"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 16v-8m0 0l-4 4m4-4l4 4"
            />
          </svg>
          <span className="text-xl font-semibold text-gray-800">ImageToCaption</span>
        </Link>
        <nav className="flex items-center gap-6">
          <Link
            href="/"
            className="text-gray-700 hover:text-gray-900 text-sm font-medium transition-colors"
          >
            Home
          </Link>
          <Link
            href="/predict"
            className="text-gray-900 text-sm font-semibold transition-colors"
          >
            Get Caption
          </Link>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8">
          {/* Upload Area - with animation */}
          <div
            className={`transition-all duration-500 ease-in-out ${
              showUploadArea
                ? "opacity-100 max-h-[500px] overflow-visible"
                : "opacity-0 max-h-0 overflow-hidden"
            }`}
          >
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging
                  ? "border-gray-400 bg-gray-50"
                  : "border-gray-300 bg-white"
              }`}
            >
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-black text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-800 transition-colors mb-4"
              >
                Upload an image
              </button>
              <p className="text-gray-600 text-sm mb-2">or drag and drop an image.</p>
              <p className="text-gray-500 text-xs">(*.jpeg and *.png images only)</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png"
                onChange={onFileChange}
                className="hidden"
              />
            </div>
          </div>

          {/* Image Preview with Cancel Button */}
          {previewUrl && !showUploadArea && (
            <div className="mt-8">
              <div className="relative inline-block">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full max-w-md h-auto rounded-lg border border-gray-200 object-contain max-h-96 shadow-sm"
                />
                {/* Cancel Button - Top Right Corner */}
                <button
                  onClick={handleCancel}
                  className="absolute top-2 right-2 bg-black/80 hover:bg-black text-white rounded-full p-2 transition-all duration-200 shadow-lg"
                  aria-label="Cancel and remove image"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>

              {/* Generate Button */}
              {!caption && !isLoading && (
                <button
                  onClick={handleGenerate}
                  className="mt-6 w-full max-w-md bg-black text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-800 transition-colors flex items-center justify-center gap-2"
                >
                  Generate Caption
                </button>
              )}

              {/* Loading State */}
              {isLoading && (
                <div className="mt-6 w-full max-w-md bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5 text-gray-700"
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
                </div>
              )}

              {/* Caption Display */}
              {caption && (
                <div className="mt-8 max-w-2xl">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Generated Caption:</h3>
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 min-h-[120px] shadow-sm">
                    <p className="text-gray-800 text-base leading-relaxed">
                      <TypingAnimation text={caption} />
                    </p>
                  </div>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="mt-6 max-w-md p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
