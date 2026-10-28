/** Only same-app paths are accepted as a redirect target after login (no open redirect). */
export function safeNext(next: string | null): string {
  const external = !next || !next.startsWith('/') || next.startsWith('//') || next.startsWith('/\\')
  return external ? '/queue' : next
}
