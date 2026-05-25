import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignSpeak AI - ASL + ArSL Translator",
  description:
    "Real-time American and Arabic Sign Language translation in your browser. Built for CSCI435 at UOWD.",
};

function Nav() {
  return (
    <nav className="sticky top-0 z-30 backdrop-blur border-b border-white/5">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
        <Link href="/" className="font-bold tracking-tight text-lg">
          <span className="gradient-text">SignSpeak</span> AI
        </Link>
        <div className="flex gap-2 text-sm">
          <Link href="/live" className="btn-secondary">Live</Link>
          <Link href="/upload" className="btn-secondary">Upload</Link>
          <Link href="/about" className="btn-secondary">About</Link>
        </div>
      </div>
    </nav>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
        <footer className="max-w-6xl mx-auto px-6 py-10 text-xs text-slate-400">
          SignSpeak AI &middot; CSCI435 Spring 2026 &middot; University of Wollongong in Dubai
        </footer>
      </body>
    </html>
  );
}
