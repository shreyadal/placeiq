// The few line icons the design uses, drawn inline so they take the text colour.

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

export function ArrowRight() {
  return (
    <svg aria-hidden width="16" height="16" viewBox="0 0 24 24" strokeWidth={2} {...stroke}>
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}

export function ChevronDown({ size = 14 }: { size?: number }) {
  return (
    <svg aria-hidden width={size} height={size} viewBox="0 0 24 24" strokeWidth={2} {...stroke}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

export function Mic() {
  return (
    <svg aria-hidden width="14" height="14" viewBox="0 0 24 24" strokeWidth={1.8} {...stroke}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}

export function Logo() {
  return (
    <svg aria-hidden width="26" height="26" viewBox="0 0 26 26" fill="none">
      <rect x="0.5" y="0.5" width="25" height="25" rx="7.5" stroke="rgba(255,255,255,0.18)" fill="rgba(255,255,255,0.04)" />
      <circle cx="13" cy="13" r="4" fill="#4C8DFF" />
    </svg>
  );
}
