export default function AboutPage() {
  return (
    <article className="max-w-3xl space-y-6">
      <h1 className="text-4xl font-bold">About SignSpeak AI</h1>

      <p className="text-slate-300 leading-relaxed">
        SignSpeak AI is a browser-based sign language translator. It captures
        hand and face movement from your webcam or from an uploaded video,
        runs a real-time computer-vision pipeline, and turns American Sign
        Language and Arabic Sign Language fingerspelling into on-screen text
        and spoken audio — no extra hardware required.
      </p>

      <section>
        <h2 className="text-2xl font-bold mb-2">User story</h2>
        <p className="text-slate-300 leading-relaxed">
          Aisha is a deaf student who often struggles to communicate with
          hearing professors during office hours when her sign-language
          interpreter is unavailable. She opens SignSpeak AI in any browser.
          The professor speaks; Aisha signs in Arabic Sign Language.
          SignSpeak AI captures her hand and face landmarks via the webcam,
          classifies each letter using a MobileNetV3 fine-tuned on ImageNet
          plus ArSL2018 ensembled with a custom-trained landmark MLP, smooths
          the predictions with a temporal voting buffer, and displays the
          translated Arabic and English text on screen, also reading it aloud
          via the browser&apos;s text-to-speech. The professor responds
          verbally, and the conversation flows.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-2">Eight computer-vision capabilities</h2>
        <ol className="list-decimal pl-6 space-y-1 text-slate-300">
          <li>Image enhancement (CLAHE)</li>
          <li>Image segmentation (MediaPipe Selfie)</li>
          <li>Binary morphological operations (erode / dilate)</li>
          <li>Keypoint detection (MediaPipe Holistic, hands + face + pose)</li>
          <li>Face detection (468 face landmarks)</li>
          <li>Edge detection (Canny on the hand crop)</li>
          <li>Video processing (optical flow + temporal voting buffer)</li>
          <li>Object recognition (MobileNetV3-Small + landmark MLP ensemble)</li>
        </ol>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-2">Open source</h2>
        <p>
          <a
            href="https://github.com/ykanjo04/SignSpeak-AI"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-300 hover:underline"
          >
            github.com/ykanjo04/SignSpeak-AI
          </a>
        </p>
      </section>
    </article>
  );
}
