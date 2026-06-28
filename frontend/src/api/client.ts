// All calls to the backend live in this module. Components never call fetch
// directly — they call these functions. That keeps the network layer in one
// place and the API contract typed.

export interface Health {
  status: string;
  app: string;
  version: string;
}

export async function getHealth(): Promise<Health> {
  const response = await fetch("/health");
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  return (await response.json()) as Health;
}