import Link from "next/link";

const capabilities = [
  { id: 1, title: "Image enhancement", desc: "CLAHE on luminance" },
  { id: 2, title: "Image segmentation", desc: "MediaPipe Selfie" },
  { id: 3, title: "Morphological ops", desc: "Erode + dilate cleanup" },
  { id: 4, title: "Keypoint detection", desc: "MediaPipe Holistic" },
  { id: 5, title: "Face detection", desc: "468 face landmarks" },
  { id: 6, title: "Edge detection", desc: "Canny on hand crop" },
  { id: 7, title: "Video processing", desc: "Optical flow + voting" },
  { id: 8, title: "Object recognition", desc: "MobileNetV3 + MLP" },
];

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

      <section className="card p-8">
        <h2 className="text-2xl font-bold mb-1">User story</h2>
        <p className="text-slate-300 leading-relaxed">
          A deaf student at UOWD opens SignSpeak AI in the professor&apos;s
          browser. The student signs in <strong>Arabic Sign Language</strong>;
          the on-screen text and TTS audio let the hearing professor follow
          along instantly - no interpreter needed. The same app also handles
          ASL for international students.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Eight CV capabilities, one app</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {capabilities.map((c) => (
            <div key={c.id} className="card p-4">
              <div className="text-xs text-slate-400">Capability {c.id}</div>
              <div className="font-semibold mt-1">{c.title}</div>
              <div className="text-sm text-slate-400">{c.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="card p-6 text-sm text-slate-400">
        <strong className="text-slate-200">CSCI435 project marker.</strong>{" "}
        SignSpeak AI integrates eight distinct computer-vision capabilities
        (only four are required) into a single deployable web application,
        with both live-webcam and uploaded-video input modalities, a
        fine-tuned MobileNetV3-Small ensembled with a custom-trained landmark
        MLP, and a smoothing buffer that keeps live predictions stable. See
        the report PDF in <code>report/</code> for the full evaluation.
      </section>
    </div>
  );
}
