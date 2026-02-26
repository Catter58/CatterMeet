import { useEffect, useRef, useState } from "react";
import { getTranscript } from "@/api";
import type { Segment } from "@/types";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Props {
  taskId: string;
  onBack: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function highlight(text: string, query: string): string {
  const safeText = escapeHtml(text);
  if (!query.trim()) return safeText;
  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return safeText.replace(
    new RegExp(`(${escapedQuery})`, "gi"),
    '<mark class="bg-yellow-200 rounded-sm">$1</mark>'
  );
}

const SPEAKER_COLORS = [
  "text-blue-700 border-blue-300",
  "text-emerald-700 border-emerald-300",
  "text-purple-700 border-purple-300",
  "text-orange-700 border-orange-300",
];

function speakerColorClass(speaker: string): string {
  const num = parseInt(speaker.replace(/\D/g, "") || "0", 10);
  return SPEAKER_COLORS[num % SPEAKER_COLORS.length];
}

export function ResultPage({ taskId, onBack }: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [query, setQuery] = useState("");
  const [filtered, setFiltered] = useState<Segment[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    getTranscript(taskId).then(({ segments }) => {
      setSegments(segments);
      setFiltered(segments);
    });
  }, [taskId]);

  useEffect(() => {
    if (!query.trim()) {
      setFiltered(segments);
      return;
    }
    const lower = query.toLowerCase();
    setFiltered(segments.filter((s) => s.text.toLowerCase().includes(lower)));
  }, [query, segments]);

  const seek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      audioRef.current.play();
    }
  };

  return (
    <div className="flex flex-col h-full gap-4">
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={onBack}>
          ← Назад
        </Button>
        <h2 className="text-xl font-semibold">Транскрипт</h2>
      </div>

      <Card className="p-4">
        <audio
          ref={audioRef}
          controls
          className="w-full"
          src={`/api/audio/${taskId}`}
        />
      </Card>

      <Input
        placeholder="Поиск по тексту..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <ScrollArea className="flex-1 border rounded-md p-4 min-h-0">
        {filtered.length === 0 && (
          <p className="text-gray-400 text-sm py-4">Ничего не найдено</p>
        )}
        {filtered.map((seg, i) => (
          <div
            key={i}
            className="mb-3 cursor-pointer hover:bg-gray-50 rounded-lg p-3 transition-colors"
            onClick={() => seek(seg.start_time)}
            title={`Перейти к ${formatTime(seg.start_time)}`}
          >
            <div className="flex items-center gap-2 mb-1">
              <Badge
                variant="outline"
                className={`text-xs font-mono ${speakerColorClass(seg.speaker)}`}
              >
                {seg.speaker}
              </Badge>
              <span className="text-xs text-gray-400 font-mono">
                {formatTime(seg.start_time)} – {formatTime(seg.end_time)}
              </span>
            </div>
            <p
              className="text-sm text-gray-800 leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: highlight(seg.text, query),
              }}
            />
          </div>
        ))}
      </ScrollArea>
    </div>
  );
}
