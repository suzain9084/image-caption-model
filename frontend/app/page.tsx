"use client";

import Link from "next/link";

export default function HomePage() {
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
            className="text-gray-900 text-sm font-semibold transition-colors"
          >
            Home
          </Link>
          <Link
            href="/predict"
            className="text-gray-700 hover:text-gray-900 text-sm font-medium transition-colors"
          >
            Get Caption
          </Link>
        </nav>
      </header>

      {/* Main Content - Centered */}
      <main className="flex-1 flex items-center justify-center px-6">
        <div className="text-center max-w-3xl">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-800 mb-6 leading-tight">
            AI-Powered<br />
            Caption Generator:<br />
            Get Inspired Today!
          </h1>
          <p className="text-gray-600 text-lg mb-8">
            Generate engaging captions effortlessly with our AI-powered caption generator. Perfect for social media posts, blog articles, and more.
          </p>
          <Link
            href="/predict"
            className="inline-block bg-black text-white px-8 py-4 rounded-lg font-medium hover:bg-gray-800 transition-colors text-lg"
          >
            Get Started
          </Link>
        </div>
      </main>
    </div>
  );
}
