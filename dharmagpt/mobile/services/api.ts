// DharmaGPT API Service Layer
// Typed client for the FastAPI backend

const BASE_URL = process.env.EXPO_PUBLIC_API_URL ?? 'https://your-beta-server.com';

// ─── Types (mirror of Python schemas.py) ─────────────────────────────────────

export type QueryMode = 'guidance' | 'story' | 'children' | 'scholar';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface QueryRequest {
  query: string;
  mode: QueryMode;
  history: ChatMessage[];
  language: string;
  filter_section?: string;
}

export interface SourceChunk {
  text: string;
  citation: string;
  section?: string;
  chapter?: number;
  verse?: number;
  score: number;
  source_type: 'text' | 'audio';
  audio_timestamp?: string;
  url?: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceChunk[];
  mode: QueryMode;
  language: string;
  query_id: string;
}

export type FeedbackRating = 'up' | 'down';

export interface FeedbackRequest {
  query_id: string;
  query: string;
  answer: string;
  mode: QueryMode;
  sources: SourceChunk[];
  rating: FeedbackRating;
  note?: string;
}

export interface HealthResponse {
  status: string;
  pinecone: boolean;
  anthropic: boolean;
  sarvam: boolean;
  vector_name: string;
}

// ─── API Client ───────────────────────────────────────────────────────────────

class DharmaAPIError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = 'DharmaAPIError';
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {}
    throw new DharmaAPIError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

// ─── Exported API calls ───────────────────────────────────────────────────────

export const dharmaApi = {
  /**
   * Ask a question — main RAG endpoint.
   */
  query(req: QueryRequest): Promise<QueryResponse> {
    return request<QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * Submit thumbs up/down feedback on a response.
   */
  feedback(req: FeedbackRequest): Promise<void> {
    return request<void>('/api/feedback', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * Health check — useful for showing connection status.
   */
  health(): Promise<HealthResponse> {
    return request<HealthResponse>('/health');
  },
};

export { DharmaAPIError };
