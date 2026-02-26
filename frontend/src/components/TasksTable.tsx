import { useEffect, useState } from "react";
import { getStatus } from "@/api";
import type { Task } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const STATUS_VARIANT: Record<
  string,
  "secondary" | "default" | "outline" | "destructive"
> = {
  pending: "secondary",
  processing: "default",
  completed: "outline",
  failed: "destructive",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "В очереди",
  processing: "Обрабатывается",
  completed: "Готово",
  failed: "Ошибка",
};

interface Props {
  taskIds: string[];
  onViewResult: (taskId: string) => void;
}

export function TasksTable({ taskIds, onViewResult }: Props) {
  const [tasks, setTasks] = useState<Record<string, Task>>({});

  useEffect(() => {
    if (taskIds.length === 0) return;

    const poll = async () => {
      for (const id of taskIds) {
        try {
          const task = await getStatus(id);
          setTasks((prev) => ({ ...prev, [id]: task }));
        } catch {
          // silently ignore polling errors
        }
      }
    };

    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, [taskIds]);

  if (taskIds.length === 0) return null;

  return (
    <div className="mt-8">
      <h2 className="text-xl font-semibold mb-4">Задачи</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Файл</TableHead>
            <TableHead>Статус</TableHead>
            <TableHead>Создано</TableHead>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {taskIds.map((id) => {
            const task = tasks[id];
            if (!task) {
              return (
                <TableRow key={id}>
                  <TableCell
                    colSpan={4}
                    className="text-gray-400 text-sm font-mono"
                  >
                    {id} — загрузка...
                  </TableCell>
                </TableRow>
              );
            }
            return (
              <TableRow key={id}>
                <TableCell className="font-mono text-sm">
                  {task.filename}
                </TableCell>
                <TableCell>
                  <Badge variant={STATUS_VARIANT[task.status] ?? "secondary"}>
                    {STATUS_LABEL[task.status] ?? task.status}
                  </Badge>
                </TableCell>
                <TableCell className="text-sm text-gray-500">
                  {new Date(task.created_at).toLocaleString("ru")}
                </TableCell>
                <TableCell>
                  {task.status === "completed" && (
                    <Button size="sm" onClick={() => onViewResult(id)}>
                      Открыть
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
