import type { Task, Segment } from "./types";

const BASE = "/api";

export async function uploadFile(file: File): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(taskId: string): Promise<Task> {
  const res = await fetch(`${BASE}/status/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTranscript(
  taskId: string
): Promise<{ segments: Segment[] }> {
  const res = await fetch(`${BASE}/transcript/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function searchTranscripts(
  q: string,
  taskId?: string
): Promise<{ results: Segment[] }> {
  const params = new URLSearchParams({ q });
  if (taskId) params.append("task_id", taskId);
  const res = await fetch(`${BASE}/search?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
