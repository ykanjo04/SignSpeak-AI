import Link from "next/link";

export default function HomePage() {
  return (
    <div className="space-y-12">
      <section className="text-center pt-10 pb-8">
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight">
          Translate <span className="gradient-text">sign language</span>
          <br />in real time.
        </h1>
        <p className="mt-6 max-w-2xl mx-auto text-lg text-slate-300">
          SignSpeak AI watches your webcam and instantly turns American Sign
          Language and Arabic Sign Language fingerspelling into text and
          spoken audio - in any browser.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <Link href="/live" className="btn-primary">Start live mode</Link>
          <Link href="/upload" className="btn-secondary">Upload a video</Link>
        </div>
      </section>
    </div>
  );
}
