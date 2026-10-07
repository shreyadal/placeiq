import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "PlaceIQ",
  description: "Adaptive AI interview coach for college placements",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <strong>PlaceIQ</strong>
          <Link href="/">New interview</Link>
          <Link href="/results">Results</Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
