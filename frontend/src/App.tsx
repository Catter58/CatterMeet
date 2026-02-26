import { useState } from "react";
import { DropZone } from "@/components/DropZone";
import { TasksTable } from "@/components/TasksTable";
import { ResultPage } from "@/components/ResultPage";

export default function App() {
  const [taskIds, setTaskIds] = useState<string[]>([]);
  const [viewingTaskId, setViewingTaskId] = useState<string | null>(null);

  const handleUploaded = (taskId: string) => {
    setTaskIds((prev) => [taskId, ...prev]);
  };

  if (viewingTaskId) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-4xl mx-auto h-[calc(100vh-3rem)] flex flex-col">
          <ResultPage
            taskId={viewingTaskId}
            onBack={() => setViewingTaskId(null)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">CatterMeet</h1>
        <p className="text-gray-500 mb-8">
          Транскрибация и диаризация аудио/видео
        </p>
        <DropZone onUploaded={handleUploaded} />
        <TasksTable taskIds={taskIds} onViewResult={setViewingTaskId} />
      </div>
    </div>
  );
}
