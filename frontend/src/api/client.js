const BASE = 'http://localhost:8000';

// ── Token helpers (localStorage) ─────────────────────────────────────────────
const getToken = () => localStorage.getItem('jwt_token');
const setToken = (t) => localStorage.setItem('jwt_token', t);
const clearToken = () => localStorage.removeItem('jwt_token');

// ── Base fetch wrapper ────────────────────────────────────────────────────────
async function request(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...options.headers };

  // Attach JWT for authenticated calls
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── Public API object ─────────────────────────────────────────────────────────
export const api = {
  /** POST /signup  →  { token, user } */
  signup: async ({ name, email, password }) => {
    const data = await request('/signup', {
      method: 'POST',
      body: JSON.stringify({ name, email, password }),
    });
    setToken(data.token);   // persist token
    return data;
  },

  /** POST /login  →  { token, user } */
  login: async ({ email, password }) => {
    const data = await request('/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    setToken(data.token);   // persist token
    return data;
  },

  /** Clear token on logout */
  logout: () => clearToken(),

  /** GET /llm?query=...&useLLM=...  –  requires JWT */
  generateJobDescription: async (prompt, useLLM = false) => {
    const params = new URLSearchParams({ query: prompt, useLLM: useLLM ? 'true' : 'false' });
    const data = await request(`/llm?${params}`);
    const jobId = 'job_' + Date.now();
    return {
      jobId,
      prompt,
      generatedMarkdown: data.model_output,
      source: data.source || (useLLM ? 'LLM' : 'SLM'),
    };
  },

  /**
   * GET /llm/stream — SSE streaming version.
   * Uses fetch (not EventSource) because EventSource can't send Authorization headers.
   * Calls onChunk(accumulatedText) on each word, onDone(fullText, source) when complete.
   */
  streamJobDescription: async (prompt, useLLM = false, { onChunk, onDone, onError } = {}) => {
    const params = new URLSearchParams({ query: prompt, useLLM: useLLM ? 'true' : 'false' });
    const token  = localStorage.getItem('jwt_token');
    const headers = { 'Accept': 'text/event-stream', 'Cache-Control': 'no-cache' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    let res;
    try {
      res = await fetch(`${BASE}/llm/stream?${params}`, { headers });
    } catch (err) {
      onError?.(err); throw err;
    }

    if (!res.ok) {
      const err = new Error((await res.json().catch(() => ({}))).detail || 'Stream failed');
      onError?.(err); throw err;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer      = '';
    let accumulated = '';
    let source      = useLLM ? 'LLM' : 'SLM';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';   // keep incomplete line for next chunk

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.done) {
            const full = data.full || accumulated;
            onDone?.(full, data.source || source);
            return { jobId: 'job_' + Date.now(), prompt, generatedMarkdown: full, source: data.source || source };
          } else {
            accumulated += data.token;
            source       = data.source || source;
            onChunk?.(accumulated);
          }
        } catch { /* skip malformed lines */ }
      }
    }
    // Fallback if stream closes without done message
    onDone?.(accumulated, source);
    return { jobId: 'job_' + Date.now(), prompt, generatedMarkdown: accumulated, source };
  },

  // POST /save-job — persists fine-tuning tuple
  saveUserEdits: async (payload) => {
    return request('/save-job', {
      method: 'POST',
      body: JSON.stringify({
        job_id:      payload.jobId,
        prompt:      payload.prompt_given_by_user || '',
        slm_output:  payload.slm_output  || '',
        llm_output:  payload.llm_output  || null,
        user_edited: payload.user_edited || null,
      }),
    });
  },

  // GET /history — last 50 jobs for logged-in user
  getHistory: async () => request('/history'),
};
