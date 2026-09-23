/** EPOCH mark: three concentric generation rings, opened like an iris, with the current epoch as a crimson node. */
export function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <g className="ring-a"><circle cx="16" cy="16" r="13.5" fill="none" stroke="#2b3040" strokeWidth="1.5" strokeDasharray="62 23" transform="rotate(-40 16 16)" /></g>
      <g className="ring-b"><circle cx="16" cy="16" r="9" fill="none" stroke="#a9b0bf" strokeWidth="1.5" strokeDasharray="40 17" transform="rotate(110 16 16)" /></g>
      <circle cx="16" cy="16" r="4.5" fill="none" stroke="#e8eaf0" strokeWidth="1.5" />
      <circle cx="26.2" cy="7.2" r="2.6" fill="#e11d48" />
      <circle cx="26.2" cy="7.2" r="4.4" fill="none" stroke="#e11d48" strokeOpacity="0.35" />
    </svg>
  );
}
