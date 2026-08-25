import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "EvoOptoDB | Organic Optoelectronic Device Database",
  description:
    "A static research browser for OLED, OFET, and OPV device records.",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
