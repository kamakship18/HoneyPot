import React from "react";

export const metadata = {
  title: "Scammer Test Interface - AI Honeypot",
  description: "Test the AI agent by acting as a scammer. Watch it extract intelligence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}


