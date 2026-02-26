import { useCallback, useState } from "react";
import { uploadFile } from "@/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Props {
  onUploaded: (taskId: string) => void;
}

export function DropZone({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        const { task_id } = await uploadFile(file);
        onUploaded(task_id);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [onUploaded]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <Card
      className={`p-12 border-2 border-dashed cursor-pointer text-center transition-colors ${
        dragging
          ? "border-blue-500 bg-blue-50"
          : "border-gray-300 hover:border-gray-400"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {uploading ? (
        <p className="text-gray-500 text-lg">Загрузка...</p>
      ) : (
        <>
          <p className="text-lg font-medium text-gray-700 mb-2">
            Перетащите аудио или видео файл сюда
          </p>
          <p className="text-sm text-gray-400 mb-4">или</p>
          <label className="cursor-pointer">
            <Button variant="outline" asChild>
              <span>Выберите файл</span>
            </Button>
            <input
              type="file"
              className="hidden"
              accept="audio/*,video/*"
              onChange={(e) =>
                e.target.files?.[0] && handleFile(e.target.files[0])
              }
            />
          </label>
        </>
      )}
      {error && <p className="text-red-500 mt-3 text-sm">{error}</p>}
    </Card>
  );
}
