/**
 * Speech input for the JARVIS screen, backed by A.D.A's own speech path.
 *
 * JARVIS listens with the browser's Web Speech API (voice.ts). Electron does not
 * ship the speech service that API needs, so A.D.A listens with its own
 * voice-activity detection (mic.ts) and transcribes on the backend (/stt).
 * This keeps the VoiceInput surface that main.ts calls.
 */

import { createMic, type Mic } from "./mic";
import { isAdaLocked, isAdaSpeaking, onAdaLockChange, requestBargeIn } from "./ada-state";
import type { VoiceInput } from "./voice";

export function createVoiceInput(
  onTranscript: (text: string) => void,
  onInterim: (text: string) => void,
  onError: (msg: string) => void,
  onMicEvent: (event: string) => void = () => {}
): VoiceInput {
  // What the screen asked for. The microphone runs only when this is true and
  // the face lock is not holding it shut.
  let wanted = false;

  const mic: Mic = createMic({
    onLevel: () => {},               // the level meter is createMicMonitor's job
    onInterim: (text) => onInterim(text),
    onBargeIn: () => requestBargeIn(),
    onUtterance: (text) => onTranscript(text),
    onBackchannel: () => {},         // no backchannel clips in this build
    onStatus: (s) => onMicEvent(s),
    onError: (msg) => onError(msg),
    isAdaSpeaking,
  });

  function run(): void {
    if (wanted && !isAdaLocked()) mic.start();
  }

  onAdaLockChange((locked) => {
    if (locked) mic.stop();
    else run();
  });

  return {
    start() {
      wanted = true;
      run();
    },
    stop() {
      wanted = false;
      mic.stop();
    },
    pause() {
      wanted = false;
      mic.stop();
    },
    resume() {
      wanted = true;
      run();
    },
    restart(reason: string) {
      onMicEvent(`restart: ${reason}`);
      mic.stop();
      window.setTimeout(run, 300);
    },
  };
}
