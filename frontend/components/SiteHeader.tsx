"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "@/components/Icons";

// Where the two background glows sit on each page, as in the design.
const GLOWS: Record<string, [React.CSSProperties, React.CSSProperties]> = {
  "/": [
    { left: "50%", top: -220, width: 1100, height: 760, marginLeft: -550 },
    { right: -200, bottom: -260, width: 700, height: 600 },
  ],
  "/interview": [
    { left: -180, top: -240, width: 1000, height: 720 },
    { right: -160, top: 120, width: 640, height: 560 },
  ],
  "/results": [
    { left: "50%", top: -260, width: 1200, height: 780, marginLeft: -600 },
    { left: -240, top: 1200, width: 760, height: 700 },
  ],
  "/model": [
    { left: "50%", top: -260, width: 1200, height: 780, marginLeft: -600 },
    { right: -240, top: 1100, width: 760, height: 700 },
  ],
};

export default function SiteHeader() {
  const pathname = usePathname();
  const onResults = pathname.startsWith("/results");
  const onModel = pathname.startsWith("/model");
  const [main, soft] = GLOWS[pathname] ?? GLOWS["/"];

  return (
    <>
      <div aria-hidden className="glow" style={main} />
      <div aria-hidden className="glow soft" style={soft} />
      <header className="site-header">
        <Link href="/" className="brand">
          <Logo />
          PlaceIQ
        </Link>
        <nav aria-label="Main" className="site-nav">
          <Link href="/" aria-current={onResults || onModel ? undefined : "page"}>
            New interview
          </Link>
          <Link href="/results" aria-current={onResults ? "page" : undefined}>
            Results
          </Link>
          <Link href="/model" aria-current={onModel ? "page" : undefined}>
            Model
          </Link>
        </nav>
      </header>
    </>
  );
}
