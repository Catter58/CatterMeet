export interface Task {
  id: string;
  filename: string;
  status: "pending" | "processing" | "completed" | "failed";
  created_at: string;
  updated_at: string;
}

export interface Segment {
  start_time: number;
  end_time: number;
  speaker: string;
  text: string;
}
