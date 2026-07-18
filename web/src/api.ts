/** Unified fetch wrapper that injects the API auth token if configured. */

const API_TOKEN: string =
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_TOKEN) || ''

function authHeaders(): Record<string, string> {
  if (!API_TOKEN) return {}
  return { Authorization: `Bearer ${API_TOKEN}` }
}

export async function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...authHeaders(),
    ...((options.headers as Record<string, string>) || {}),
  }
  return fetch(url, { ...options, headers })
}

export async function apiPost(url: string, body?: unknown): Promise<Response> {
  return apiFetch(url, {
    method: 'POST',
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
}

export async function apiGet(url: string): Promise<Response> {
  return apiFetch(url, { method: 'GET' })
}
