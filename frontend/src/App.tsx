import { useEffect, useState } from "react";
import { Room, RoomEvent } from "livekit-client";
import { RoomAudioRenderer, RoomContext } from "@livekit/components-react";
import "@livekit/components-styles";

const TOKEN_URL = "http://localhost:8001/token";

export default function App() {
  const [room] = useState(() => new Room({ adaptiveStream: true, dynacast: true }));
  const [status, setStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [agentJoined, setAgentJoined] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkAgent = () => {
      const hasAgent = Array.from(room.remoteParticipants.values()).some((p) => p.isAgent);
      setAgentJoined(hasAgent);
    };

    room.on(RoomEvent.ParticipantConnected, checkAgent);
    room.on(RoomEvent.ParticipantDisconnected, checkAgent);
    room.on(RoomEvent.Disconnected, () => {
      setStatus("disconnected");
      setAgentJoined(false);
      setMicOn(false);
    });

    return () => {
      room.removeAllListeners();
      room.disconnect();
    };
  }, [room]);

  async function connect() {
    try {
      setError(null);
      setStatus("connecting");

      // 1. Token server se token lo
      const res = await fetch(TOKEN_URL);
      if (!res.ok) throw new Error("Token server se token nahi mila");
      const { token, url } = await res.json();

      // 2. LiveKit room join karo
      await room.connect(url, token);

      // 3. Mic on karo
      await room.localParticipant.setMicrophoneEnabled(true);
      setMicOn(true);
      setStatus("connected");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed");
      setStatus("disconnected");
    }
  }

  function disconnect() {
    room.disconnect();
  }

  async function toggleMic() {
    const next = !micOn;
    await room.localParticipant.setMicrophoneEnabled(next);
    setMicOn(next);
  }

  return (
    <RoomContext.Provider value={room}>
      <RoomAudioRenderer />

      <div style={{ maxWidth: 480, margin: "40px auto", fontFamily: "sans-serif", textAlign: "center" }}>
        <h2>Voice Search Assistant</h2>

        <p>Status: <b>{status}</b></p>
        <p>Agent: <b>{agentJoined ? "joined" : "not joined"}</b></p>

        {status !== "connected" ? (
          <button onClick={connect} disabled={status === "connecting"}>
            {status === "connecting" ? "Connecting..." : "Start conversation"}
          </button>
        ) : (
          <>
            <button onClick={toggleMic}>{micOn ? "Mute mic" : "Unmute mic"}</button>{" "}
            <button onClick={disconnect}>End</button>
          </>
        )}

        {error && <p style={{ color: "red" }}>{error}</p>}
      </div>
    </RoomContext.Provider>
  );
}